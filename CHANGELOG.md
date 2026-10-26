# Changelog

## 0.2.1

### Changed

- Removed the 4096-token default for `--max-tokens`; omitted API `max_tokens` and pi overlay `maxTokens` when unset, preserving backend/server defaults. Reserved output headroom in context gating only for explicit budgets; unset budgets provided no headroom guarantee.

### Fixed

- Prevented orphaned CLI requests after interruption or exceptional exits by killing active process groups, reaping children, joining output readers, and closing streams.

### Tests

- Added real-subprocess checks for interruption after spawn and during output processing, normal exits, and timeouts.
