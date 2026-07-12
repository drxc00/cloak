import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';

type Provider = 'openai' | 'anthropic' | 'deepseek' | 'groq';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY!;
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY!;
const GROQ_API_KEY = process.env.GROQ_API_KEY!;

const clients = {
  openai: new OpenAI({
    apiKey: OPENAI_API_KEY,
  }),
  deepseek: new OpenAI({
    apiKey: DEEPSEEK_API_KEY,
    baseURL: 'https://api.deepseek.com/v1',
  }),
  groq: new OpenAI({
    apiKey: GROQ_API_KEY,
    baseURL: 'https://api.groq.com/openai/v1',
  }),
};

const anthropicClient = new Anthropic({
  apiKey: ANTHROPIC_API_KEY,
});

const FALLBACK_CHAIN: Provider[] = ['openai', 'anthropic', 'deepseek', 'groq'];

export async function complete(
  prompt: string,
  preferredProvider?: Provider
): Promise<string> {
  const providers = preferredProvider
    ? [preferredProvider]
    : FALLBACK_CHAIN;

  for (const provider of providers) {
    try {
      console.log(`[ai] Trying provider: ${provider}`);

      switch (provider) {
        case 'openai':
        case 'deepseek':
        case 'groq': {
          const model = provider === 'groq' ? 'llama3-70b-8192' : 'gpt-4o';
          const response = await clients[provider].chat.completions.create({
            model,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: 1024,
          });
          return response.choices[0].message.content ?? '';
        }
        case 'anthropic': {
          const response = await anthropicClient.messages.create({
            model: 'claude-3-5-sonnet-20241022',
            max_tokens: 1024,
            messages: [{ role: 'user', content: prompt }],
          });
          const block = response.content[0];
          return block.type === 'text' ? block.text : '';
        }
      }
    } catch (err) {
      console.error(`[ai] Provider ${provider} failed:`, err);
      continue;
    }
  }

  throw new Error('[ai] All providers exhausted — no fallback available');
}
