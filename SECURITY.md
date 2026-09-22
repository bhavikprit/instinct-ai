# Security Policy

## Supported Versions

We release security updates and bug fixes for the latest active release versions of `instinct-ai` (and `reflex` backward-compatibility layer).

| Version | Supported          |
| :------ | :----------------- |
| 0.2.x   | :white_check_mark: |
| < 0.2.0 | :x:                |

---

## Reporting a Vulnerability

The Instinct AI / Reflex-AI team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose findings.

### Private Reporting Channels
If you discover a security vulnerability (such as bypasses in guardrails, memory buffer safety in the C runtime, or authentication issues in the Envoy Gateway), **please do not create a public GitHub issue**.

Instead, please report the vulnerability privately via one of the following methods:

1. **GitHub Private Vulnerability Reporting**:
   - Go to the [Security Advisories](https://github.com/bhavikprit/reflex-ai/security/advisories) tab of this repository.
   - Click **"Report a vulnerability"** to open a confidential report.

2. **Direct Security Contact**:
   - Email: **bhavikpatel13792@gmail.com**
   - Please include:
     - Component name (`instinct-ai`, `reflex`, `reflex_c`, `gateway`, or `guardrails`).
     - Clear description of the vulnerability and attack vector.
     - Reproducible proof-of-concept (PoC) script or curl command.
     - Potential impact (e.g. DoS, memory corruption, guardrail bypass, data leakage).

---

## Response Timeline & SLA

* **Initial Acknowledgment**: Within **24–48 hours**.
* **Vulnerability Assessment & Triage**: Within **5 business days**.
* **Fix & Release Coordination**: We will work with you to test the fix and coordinate a public release along with CVE assignment if applicable.

---

## Security Philosophy

* **Zero Third-Party Dependencies**: The core Python package has zero external third-party dependencies, drastically minimizing supply-chain attack vectors.
* **Deterministic Verification**: Binary formats (`.reflex-compile`, `.reflex-index`, `.reflex-cascade`, `.reflex-kv`) enforce 32-bit CRC32 checksums and strict magic header validation to prevent arbitrary code execution or deserialization exploits.
* **Cryptographic Tamper-Evidence**: Policy and audit logs employ cryptographic Merkle Trees to provide immutable mathematical proof of execution.
