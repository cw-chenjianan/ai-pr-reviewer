import './fetch-polyfill'

import {info, setFailed, warning} from '@actions/core'
import {
  ChatGPTAPI,
  ChatGPTError,
  ChatMessage,
  SendMessageOptions
  // eslint-disable-next-line import/no-unresolved
} from 'chatgpt'
import pRetry from 'p-retry'
import {OpenAIOptions, Options} from './options'

// define type to save parentMessageId and conversationId
export interface Ids {
  parentMessageId?: string
  conversationId?: string
}

export class Bot {
  private readonly api: ChatGPTAPI | null = null // not free
  private readonly useDirectFetch: boolean = false // 是否直接使用 fetch（用于 Azure OpenAI）
  private readonly directFetchUrl: string = ''

  private readonly options: Options
  private readonly openaiOptions: OpenAIOptions

  constructor(options: Options, openaiOptions: OpenAIOptions) {
    this.options = options
    this.openaiOptions = openaiOptions
    if (process.env.OPENAI_API_KEY) {
      const currentDate = new Date().toISOString().split('T')[0]
      const systemMessage = `${options.systemMessage} 
Knowledge cutoff: ${openaiOptions.tokenLimits.knowledgeCutOff}
Current date: ${currentDate}

IMPORTANT: Entire response must be in the language with ISO code: ${options.language}
`

      // 如果配置了自定义请求头，直接使用 fetch API，不使用 chatgpt 库
      if (process.env.AZURE_OPENAI_API_KEY) {
        info('Using direct fetch API for Azure OpenAI (bypassing chatgpt library)')
        this.useDirectFetch = true
        this.directFetchUrl = options.apiBaseUrl
        info(`Direct fetch URL: ${this.directFetchUrl}`)
        info(`Custom headers: ${JSON.stringify(Object.keys(options.customHeaders))}`)
      } else {
        // 使用标准的 chatgpt 库
        const apiConfig: any = {
          apiBaseUrl: options.apiBaseUrl,
          systemMessage,
          apiKey: process.env.OPENAI_API_KEY,
          apiOrg: process.env.OPENAI_API_ORG ?? undefined,
          debug: options.debug,
          maxModelTokens: openaiOptions.tokenLimits.maxTokens,
          maxResponseTokens: openaiOptions.tokenLimits.responseTokens,
          completionParams: {
            temperature: options.openaiModelTemperature,
            model: openaiOptions.model
          }
        }

        this.api = new ChatGPTAPI(apiConfig)
      }
    } else {
      const err =
        "Unable to initialize the OpenAI API, both 'OPENAI_API_KEY' environment variable are not available"
      throw new Error(err)
    }
  }

  chat = async (message: string, ids: Ids): Promise<[string, Ids]> => {
    let res: [string, Ids] = ['', {}]
    try {
      res = await this.chat_(message, ids)
      return res
    } catch (e: unknown) {
      if (e instanceof ChatGPTError) {
        warning(`Failed to chat: ${e}, backtrace: ${e.stack}`)
      }
      return res
    }
  }

  private readonly chat_ = async (
    message: string,
    ids: Ids
  ): Promise<[string, Ids]> => {
    // record timing
    const start = Date.now()
    if (!message) {
      return ['', {}]
    }

    let response: ChatMessage | undefined

    // 如果使用直接 fetch（Azure OpenAI）
    if (this.useDirectFetch) {
      try {
        response = await this.directFetchRequest(message, ids)
      } catch (e: unknown) {
        if (e instanceof Error) {
          info(
            `response: ${response}, failed to send message to openai: ${e}, backtrace: ${e.stack}`
          )
        }
      }
    } else if (this.api != null) {
      // 使用 chatgpt 库
      const opts: SendMessageOptions = {
        timeoutMs: this.options.openaiTimeoutMS
      }
      if (ids.parentMessageId) {
        opts.parentMessageId = ids.parentMessageId
      }
      try {
        response = await pRetry(() => this.api!.sendMessage(message, opts), {
          retries: this.options.openaiRetries
        })
      } catch (e: unknown) {
        if (e instanceof ChatGPTError) {
          info(
            `response: ${response}, failed to send message to openai: ${e}, backtrace: ${e.stack}`
          )
        }
      }
    } else {
      setFailed('The OpenAI API is not initialized')
    }
    
    const end = Date.now()
    info(`response: ${JSON.stringify(response)}`)
    info(
      `openai sendMessage (including retries) response time: ${
        end - start
      } ms`
    )
    
    let responseText = ''
    if (response != null) {
      responseText = response.text
    } else {
      warning('openai response is null')
    }
    // remove the prefix "with " in the response
    if (responseText.startsWith('with ')) {
      responseText = responseText.substring(5)
    }
    if (this.options.debug) {
      info(`openai responses: ${responseText}`)
    }
    const newIds: Ids = {
      parentMessageId: response?.id,
      conversationId: response?.conversationId
    }
    return [responseText, newIds]
  }

  private readonly directFetchRequest = async (
    message: string,
    ids: Ids
  ): Promise<ChatMessage> => {
    // 构建请求头：直接使用环境变量中的 OPENAI_API_KEY
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'api-key': process.env.AZURE_OPENAI_API_KEY || '',
    }
    
    if (this.options.debug) {
      info(`Using api-key from env.AZURE_OPENAI_API_KEY`)
      info(`Additional custom headers: ${JSON.stringify(Object.keys(this.options.customHeaders))}`)
    }

    const body = {
      messages: [
        {
          role: 'system',
          content: this.options.systemMessage
        },
        {
          role: 'user',
          content: message
        }
      ],
      temperature: this.options.openaiModelTemperature,
      max_tokens: this.openaiOptions.tokenLimits.responseTokens
    }

    if (this.options.debug) {
      info(`Direct fetch request to: ${this.directFetchUrl}`)
      info(`Headers: ${JSON.stringify(Object.keys(headers))}`)
      info(`Body: ${JSON.stringify(body)}`)
    }

    const response = await fetch(this.directFetchUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(body)
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(
        `Azure OpenAI error ${response.status}: ${errorText}`
      )
    }

    const data = await response.json()

    if (this.options.debug) {
      info(`Azure OpenAI response: ${JSON.stringify(data)}`)
    }

    // 转换为 ChatMessage 格式
    const chatMessage: ChatMessage = {
      id: data.id || `msg_${Date.now()}`,
      text: data.choices?.[0]?.message?.content || '',
      role: 'assistant',
      parentMessageId: ids.parentMessageId,
      conversationId: ids.conversationId || `conv_${Date.now()}`
    }

    return chatMessage
  }
}
