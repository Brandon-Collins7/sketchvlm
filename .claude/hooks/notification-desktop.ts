#!/usr/bin/env npx tsx
/* eslint-disable node/prefer-global/process */
/**
 * Claude Code Notification Hook - Desktop Alerts
 *
 * Sends system notifications when Claude needs attention:
 * - Permission prompts
 * - Idle prompts (waiting for input)
 */

import type { NotificationHookInput } from '@anthropic-ai/claude-agent-sdk'

import { execSync } from 'node:child_process'
import { readFileSync } from 'node:fs'

function readStdin(): string {
  return readFileSync(0, 'utf-8')
}

function sendMacNotification(title: string, message: string): void {
  // Escape special characters for AppleScript
  const escapedTitle = title.replace(/"/g, '\\"')
  const escapedMessage = message.replace(/"/g, '\\"')

  const script = `display notification "${escapedMessage}" with title "${escapedTitle}" sound name "Ping"`

  try {
    execSync(`osascript -e '${script}'`, { stdio: 'ignore' })
  }
  catch {
    // Notification failed, ignore silently
  }
}

function main(): void {
  const rawInput = readStdin()

  let parsedInput: unknown
  try {
    parsedInput = JSON.parse(rawInput)
  }
  catch {
    process.exit(0)
  }

  const input = parsedInput as NotificationHookInput

  const notificationType = (input as { notification_type?: string }).notification_type
  const message = input.message

  switch (notificationType) {
    case 'permission_prompt':
      sendMacNotification('Claude Code - Permission Required', message || 'Claude needs your permission to continue')
      break
    case 'idle_prompt':
      sendMacNotification('Claude Code - Waiting', message || 'Claude is waiting for your input')
      break
    default:
      // Don't notify for other types
      break
  }

  process.exit(0)
}

main()
