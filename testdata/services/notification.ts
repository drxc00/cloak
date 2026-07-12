import axios from 'axios';

interface NotificationPayload {
  channel: string;
  message: string;
  attachments?: Array<{ title: string; text: string; color: string }>;
}

const SLACK_WEBHOOK_URL = process.env.SLACK_WEBHOOK_URL as string;
const SLACK_BOT_TOKEN = process.env.SLACK_BOT_TOKEN as string;
const SENDGRID_API_KEY = process.env.SENDGRID_API_KEY as string;

export async function sendSlackNotification(payload: NotificationPayload): Promise<void> {
  if (!SLACK_WEBHOOK_URL) {
    throw new Error('SLACK_WEBHOOK_URL not configured');
  }

  try {
    await axios.post(`https://${SLACK_WEBHOOK_URL}`, {
      channel: payload.channel,
      text: payload.message,
      attachments: payload.attachments,
    });

    console.log(`[notification] Sent Slack message to ${payload.channel}`);
  } catch (err) {
    console.error('[notification] Slack webhook failed, trying bot API');
    if (SLACK_BOT_TOKEN) {
      await axios.post(
        'https://slack.com/api/chat.postMessage',
        {
          channel: payload.channel,
          text: payload.message,
          attachments: payload.attachments,
        },
        {
          headers: {
            Authorization: `Bearer ${SLACK_BOT_TOKEN}`,
            'Content-Type': 'application/json',
          },
        }
      );
    }
  }
}

export async function sendEmail(to: string, subject: string, body: string): Promise<void> {
  if (!SENDGRID_API_KEY) {
    throw new Error('SENDGRID_API_KEY not configured');
  }

  await axios.post(
    'https://api.sendgrid.com/v3/mail/send',
    {
      personalizations: [{ to: [{ email: to }] }],
      from: { email: 'noreply@company.com' },
      subject,
      content: [{ type: 'text/plain', value: body }],
    },
    {
      headers: {
        Authorization: `Bearer ${SENDGRID_API_KEY}`,
        'Content-Type': 'application/json',
      },
    }
  );

  console.log(`[notification] Sent email to ${to}`);
}

// Example usage (DO NOT UNCOMMENT IN PROD):
// sendSlackNotification({
//   channel: '#alerts',
//   message: 'Deploy completed successfully',
// });
