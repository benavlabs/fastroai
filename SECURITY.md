# Security Policy

## Supported Versions

FastroAI is currently in pre-1.0.0 development. During this phase, only the latest version receives security updates and patches.

| Version        | Supported          |
| -------------- | ------------------ |
| Latest Release | :white_check_mark: |
| Older Versions | :x:                |

We strongly recommend always using the latest version of FastroAI to ensure you have all security fixes and improvements.

## Reporting a Vulnerability

We take the security of FastroAI seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Reporting Process

1. **Do Not** disclose the vulnerability publicly until it has been addressed by our team
2. Submit the vulnerability report through one of these channels:

   - Email: contact+fastroai@benav.io
   - GitHub Security Advisory: https://github.com/benavlabs/fastroai/security/advisories/new

### What to Include

Please provide detailed information about the vulnerability, including:

- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if available)
- Your contact information for follow-up questions

### Response Timeline

- Initial Response: Within 48 hours
- Status Update: Within 1 week
- Fix Timeline: Based on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Within 60 days

### What to Expect

1. **Acknowledgment**: You will receive an acknowledgment of your report within 48 hours
2. **Investigation**: Our team will investigate the issue and determine its impact
3. **Updates**: You will receive updates on the status of your report
4. **Resolution**: Once resolved, you will be notified of the fix
5. **Public Disclosure**: Coordinated disclosure after the fix is released

## Security Considerations

### API Key Security

FastroAI interacts with AI providers (OpenAI, Anthropic, etc.). When using FastroAI, ensure:

1. API keys are stored securely (environment variables, secrets manager)
2. API keys are never committed to version control
3. API keys have appropriate scope and permissions
4. Key rotation policies are in place

### Cost Control

FastroAI tracks costs in microcents. Implement:

1. Cost budgets to prevent runaway spending
2. Rate limiting on your application layer
3. Monitoring and alerting for unusual usage patterns

### Prompt Injection

When building AI applications with FastroAI:

1. Validate and sanitize user inputs
2. Use system prompts to constrain agent behavior
3. Implement output validation where appropriate
4. Consider using content filtering

### Data Protection

1. Never expose sensitive data in prompts or logs
2. Implement proper logging practices (redact PII)
3. Use HTTPS for all API communications
4. Follow data protection regulations (GDPR, CCPA, etc.)

## Best Practices

1. **Always use the latest supported version**
2. Implement proper authentication and authorization in your application
3. Use HTTPS for all API endpoints
4. Regularly update dependencies
5. Follow the principle of least privilege for API keys
6. Implement proper error handling (don't leak internal details)
7. Use secure configuration management
8. Regular security audits and testing

## Security Features

FastroAI includes several security-conscious features:

1. **Cost Tracking**: Monitor and limit API spending
2. **Timeout Controls**: Prevent runaway operations via `@safe_tool`
3. **Error Handling**: Graceful failures that don't expose internals
4. **Tracing**: Audit trail for AI operations

## Disclaimer

While FastroAI implements security best practices, it's crucial to properly secure your application as a whole. This includes:

1. Proper authentication implementation
2. Authorization controls
3. Input validation
4. Error handling
5. Secure configuration
6. Regular security updates
7. Monitoring and logging

## Updates and Notifications

Stay informed about security updates:

1. Watch the GitHub repository
2. Follow our security announcements
3. Monitor our release notes

## License

This security policy is part of the FastroAI project and is subject to the same license terms.
