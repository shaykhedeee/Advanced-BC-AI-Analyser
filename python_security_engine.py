"""
python_security_engine.py — Security & Education Module for Edge Tracker
Provides Python security tools, integrity monitoring, and educational features.

Features:
  - File integrity monitoring (SHA-256 checksums)
  - Password strength analysis & hash cracking education
  - Port scanning (educational Kali-style)
  - Data integrity verification for game data
  - Encryption/decryption utilities (Fernet AES)
  - HTTP vulnerability scanner (educational)
  - Code security audit patterns
  - Python security education mode
"""

import os
import hashlib
import hmac
import json
import time
import re
import socket
import struct
import secrets
from typing import Optional, Dict, List, Any
from collections import Counter

# Optional imports
try:
    from cryptography.fernet import Fernet
    _FERNET = True
except ImportError:
    _FERNET = False

try:
    import bcrypt as _bcrypt
    _BCRYPT = True
except ImportError:
    _BCRYPT = False


class SecurityEngine:
    """Education-focused security engine with practical Python tools."""

    def __init__(self):
        self.integrity_db = {}  # path -> hash
        self.scan_results = []
        self.education_topics = self._build_education_library()

    # ==================================================================
    # 1. HASHING & INTEGRITY
    # ==================================================================
    def hash_data(self, data: str, algorithm: str = "sha256") -> str:
        """Hash a string using the specified algorithm."""
        algos = {
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "blake2b": hashlib.blake2b,
        }
        func = algos.get(algorithm, hashlib.sha256)
        return func(data.encode()).hexdigest()

    def hmac_sign(self, message: str, key: str, algorithm: str = "sha256") -> str:
        """Create an HMAC signature (used in provably fair verification)."""
        return hmac.new(
            key.encode(), message.encode(), getattr(hashlib, algorithm)
        ).hexdigest()

    def verify_hmac(self, message: str, key: str, expected_sig: str) -> bool:
        """Verify an HMAC signature."""
        actual = self.hmac_sign(message, key)
        return hmac.compare_digest(actual, expected_sig)

    def hash_file(self, filepath: str) -> str:
        """SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def monitor_files(self, directory: str, extensions: tuple = (".py", ".json", ".txt")):
        """Build an integrity database for files in a directory."""
        count = 0
        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.endswith(extensions):
                    path = os.path.join(root, fname)
                    self.integrity_db[path] = self.hash_file(path)
                    count += 1
        return count

    def check_integrity(self) -> Dict[str, str]:
        """Check if any monitored files have changed."""
        changes = {}
        for path, expected_hash in self.integrity_db.items():
            if not os.path.exists(path):
                changes[path] = "DELETED"
            else:
                current = self.hash_file(path)
                if current != expected_hash:
                    changes[path] = "MODIFIED"
        return changes

    # ==================================================================
    # 2. DATA INTEGRITY (for game data)
    # ==================================================================
    def verify_data_integrity(self, data: list) -> dict:
        """
        Check game data for statistical anomalies that might indicate
        tampering or manipulation.
        """
        import numpy as np
        if len(data) < 10:
            return {"status": "insufficient_data", "n": len(data)}

        arr = np.array(data, dtype=float)
        results = {
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "checks": {},
        }

        # Check 1: NIST frequency test — are values uniformly distributed?
        median = np.median(arr)
        above = (arr > median).sum()
        below = (arr <= median).sum()
        balance = min(above, below) / max(above, below)
        results["checks"]["frequency_balance"] = {
            "above_median": int(above),
            "below_median": int(below),
            "balance_ratio": round(balance, 3),
            "pass": balance > 0.8,
        }

        # Check 2: Runs test — are there suspicious streaks?
        runs = 1
        for i in range(1, len(arr)):
            if (arr[i] > median) != (arr[i - 1] > median):
                runs += 1
        expected_runs = 2 * above * below / len(arr) + 1
        runs_ratio = runs / max(expected_runs, 1)
        results["checks"]["runs_test"] = {
            "actual_runs": runs,
            "expected_runs": round(expected_runs, 1),
            "ratio": round(runs_ratio, 3),
            "pass": 0.7 < runs_ratio < 1.3,
        }

        # Check 3: Entropy check
        hist, _ = np.histogram(arr, bins=20)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(20)
        results["checks"]["entropy"] = {
            "entropy": round(entropy, 3),
            "max_possible": round(max_entropy, 3),
            "ratio": round(entropy / max_entropy, 3),
            "pass": entropy / max_entropy > 0.6,
        }

        # Overall verdict
        passed = sum(1 for c in results["checks"].values() if c["pass"])
        total = len(results["checks"])
        results["verdict"] = "OK" if passed == total else f"WARN ({total - passed} checks failed)"
        results["score"] = f"{passed}/{total}"

        return results

    # ==================================================================
    # 3. PASSWORD ANALYSIS (Educational)
    # ==================================================================
    def analyze_password(self, password: str) -> dict:
        """Analyze password strength — educational tool."""
        result = {
            "length": len(password),
            "has_upper": bool(re.search(r"[A-Z]", password)),
            "has_lower": bool(re.search(r"[a-z]", password)),
            "has_digit": bool(re.search(r"\d", password)),
            "has_special": bool(re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)),
            "common_patterns": [],
        }

        # Check common patterns
        patterns = [
            (r"123|abc|qwerty|password|admin", "common sequence"),
            (r"(.)\1{2,}", "repeated characters"),
            (r"(..+?)\1+", "repeated pattern"),
            (r"^[a-zA-Z]+$", "letters only"),
            (r"^\d+$", "digits only"),
        ]
        for pat, name in patterns:
            if re.search(pat, password, re.IGNORECASE):
                result["common_patterns"].append(name)

        # Score
        score = 0
        if result["length"] >= 8:
            score += 1
        if result["length"] >= 12:
            score += 1
        if result["length"] >= 16:
            score += 1
        for k in ("has_upper", "has_lower", "has_digit", "has_special"):
            if result[k]:
                score += 1
        if not result["common_patterns"]:
            score += 1

        # Entropy estimate
        charset_size = 0
        if result["has_lower"]:
            charset_size += 26
        if result["has_upper"]:
            charset_size += 26
        if result["has_digit"]:
            charset_size += 10
        if result["has_special"]:
            charset_size += 32
        if charset_size == 0:
            charset_size = 26
        import math
        result["entropy_bits"] = round(math.log2(charset_size) * len(password), 1)

        result["score"] = score
        result["rating"] = (
            "Very Weak" if score <= 2 else
            "Weak" if score <= 3 else
            "Moderate" if score <= 5 else
            "Strong" if score <= 6 else
            "Very Strong"
        )
        return result

    def generate_password(self, length: int = 20) -> str:
        """Generate a cryptographically secure random password."""
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def hash_password(self, password: str) -> str:
        """Hash a password with bcrypt (or SHA-256 fallback)."""
        if _BCRYPT:
            return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against a hash."""
        if _BCRYPT and hashed.startswith("$2"):
            return _bcrypt.checkpw(password.encode(), hashed.encode())
        return hashlib.sha256(password.encode()).hexdigest() == hashed

    # ==================================================================
    # 4. ENCRYPTION UTILITIES
    # ==================================================================
    def generate_key(self) -> str:
        """Generate a Fernet encryption key."""
        if _FERNET:
            return Fernet.generate_key().decode()
        return secrets.token_hex(32)

    def encrypt(self, data: str, key: str) -> str:
        """Encrypt data with Fernet AES."""
        if _FERNET:
            f = Fernet(key.encode())
            return f.encrypt(data.encode()).decode()
        # Fallback: XOR with key hash (NOT secure, just educational)
        key_hash = hashlib.sha256(key.encode()).digest()
        encrypted = bytes(b ^ key_hash[i % len(key_hash)]
                         for i, b in enumerate(data.encode()))
        return encrypted.hex()

    def decrypt(self, encrypted_data: str, key: str) -> str:
        """Decrypt data."""
        if _FERNET:
            f = Fernet(key.encode())
            return f.decrypt(encrypted_data.encode()).decode()
        key_hash = hashlib.sha256(key.encode()).digest()
        data_bytes = bytes.fromhex(encrypted_data)
        decrypted = bytes(b ^ key_hash[i % len(key_hash)]
                         for i, b in enumerate(data_bytes))
        return decrypted.decode()

    # ==================================================================
    # 5. PORT SCANNER (Educational — Kali-style)
    # ==================================================================
    def scan_port(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """Check if a single port is open."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def scan_ports(self, host: str = "127.0.0.1",
                   ports: Optional[List[int]] = None,
                   timeout: float = 0.5) -> Dict[int, str]:
        """
        Scan common ports on a host (educational — like nmap lite).
        Only scans localhost by default for safety.
        """
        if ports is None:
            ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445,
                     993, 995, 3306, 3389, 5432, 5900, 6379, 8080, 8443, 9090]

        service_map = {
            21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
            80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 445: "SMB",
            993: "IMAPS", 995: "POP3S", 3306: "MySQL", 3389: "RDP",
            5432: "PostgreSQL", 5900: "VNC", 6379: "Redis", 8080: "HTTP-Alt",
            8443: "HTTPS-Alt", 9090: "Prometheus",
        }

        results = {}
        for port in ports:
            if self.scan_port(host, port, timeout):
                service = service_map.get(port, "unknown")
                results[port] = f"OPEN ({service})"

        self.scan_results.append({
            "host": host, "time": time.time(),
            "open_ports": list(results.keys()),
        })
        return results

    # ==================================================================
    # 6. CODE SECURITY PATTERNS
    # ==================================================================
    def audit_code_string(self, code: str) -> List[dict]:
        """
        Static analysis: check code for common security issues.
        Returns a list of findings.
        """
        findings = []
        patterns = [
            (r"eval\s*\(", "CRITICAL", "Use of eval() — potential code injection"),
            (r"exec\s*\(", "CRITICAL", "Use of exec() — potential code injection"),
            (r"pickle\.loads?", "HIGH", "Pickle deserialization — arbitrary code execution risk"),
            (r"os\.system\s*\(", "HIGH", "os.system() — use subprocess.run instead"),
            (r"subprocess.*shell\s*=\s*True", "HIGH", "Shell=True — command injection risk"),
            (r"__import__\s*\(", "MEDIUM", "Dynamic import — review carefully"),
            (r"password\s*=\s*['\"][^'\"]+['\"]", "HIGH", "Hardcoded password detected"),
            (r"api_key\s*=\s*['\"][^'\"]+['\"]", "HIGH", "Hardcoded API key detected"),
            (r"secret\s*=\s*['\"][^'\"]+['\"]", "HIGH", "Hardcoded secret detected"),
            (r"SELECT.*FROM.*\+", "HIGH", "Possible SQL injection (string concat)"),
            (r"f['\"].*SELECT", "MEDIUM", "Possible SQL injection (f-string)"),
            (r"\.format\(.*SELECT", "MEDIUM", "Possible SQL injection (.format)"),
            (r"random\.random\(\)|random\.randint", "LOW", "Non-cryptographic RNG — use secrets for security"),
            (r"verify\s*=\s*False", "MEDIUM", "SSL verification disabled"),
            (r"debug\s*=\s*True", "LOW", "Debug mode enabled"),
            (r"assert\s+", "LOW", "Assert statements removed in optimized mode"),
        ]

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern, severity, message in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "line": i,
                        "severity": severity,
                        "message": message,
                        "code": stripped[:80],
                    })

        return findings

    def audit_file(self, filepath: str) -> List[dict]:
        """Audit a Python file for security issues."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        findings = self.audit_code_string(code)
        for finding in findings:
            finding["file"] = filepath
        return findings

    def audit_directory(self, directory: str) -> List[dict]:
        """Audit all Python files in a directory."""
        all_findings = []
        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.endswith(".py"):
                    path = os.path.join(root, fname)
                    all_findings.extend(self.audit_file(path))
        return all_findings

    # ==================================================================
    # 7. PYTHON SECURITY EDUCATION
    # ==================================================================
    def _build_education_library(self) -> dict:
        return {
            "hashing": {
                "title": "Cryptographic Hashing (SHA-256, bcrypt)",
                "description": (
                    "Hashing transforms data into a fixed-size fingerprint. "
                    "SHA-256 is used for data integrity (provably fair games). "
                    "bcrypt is used for password storage with salt."
                ),
                "example": (
                    "import hashlib\n"
                    "h = hashlib.sha256(b'my data').hexdigest()\n"
                    "# For passwords, use bcrypt:\n"
                    "import bcrypt\n"
                    "hashed = bcrypt.hashpw(b'password', bcrypt.gensalt())"
                ),
            },
            "hmac": {
                "title": "HMAC — Message Authentication Codes",
                "description": (
                    "HMAC combines a secret key with a hash to verify both "
                    "integrity AND authenticity. Used in provably fair games "
                    "to verify outcomes weren't manipulated."
                ),
                "example": (
                    "import hmac, hashlib\n"
                    "sig = hmac.new(b'server_seed', b'client_seed:nonce',\n"
                    "               hashlib.sha256).hexdigest()\n"
                    "# Convert to crash point:\n"
                    "h = int(sig[:8], 16)\n"
                    "crash_point = max(1.00, (2**32) / (h + 1) * 0.99)"
                ),
            },
            "encryption": {
                "title": "Symmetric Encryption (AES/Fernet)",
                "description": (
                    "Encryption makes data unreadable without the key. "
                    "Fernet uses AES-128-CBC with HMAC authentication."
                ),
                "example": (
                    "from cryptography.fernet import Fernet\n"
                    "key = Fernet.generate_key()\n"
                    "f = Fernet(key)\n"
                    "encrypted = f.encrypt(b'secret data')\n"
                    "decrypted = f.decrypt(encrypted)"
                ),
            },
            "input_validation": {
                "title": "Input Validation & Sanitization",
                "description": (
                    "Never trust user input. Validate types, ranges, lengths. "
                    "Sanitize for SQL injection, XSS, path traversal."
                ),
                "example": (
                    "import re, html\n"
                    "def sanitize(text):\n"
                    "    text = html.escape(text)\n"
                    "    text = re.sub(r'[<>\"\\']', '', text)\n"
                    "    return text[:1000]  # length limit\n\n"
                    "# SQL: use parameterized queries\n"
                    "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
                ),
            },
            "port_scanning": {
                "title": "Network Port Scanning (Nmap-style)",
                "description": (
                    "Port scanning checks which network services are running. "
                    "Used in penetration testing to find attack surfaces. "
                    "Python's socket module can do basic TCP connect scans."
                ),
                "example": (
                    "import socket\n"
                    "def scan(host, port):\n"
                    "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                    "    s.settimeout(1)\n"
                    "    result = s.connect_ex((host, port))\n"
                    "    s.close()\n"
                    "    return result == 0  # True = open"
                ),
            },
            "randomness_testing": {
                "title": "Testing RNG Quality (NIST-style)",
                "description": (
                    "In provably fair games, we need to verify RNG isn't biased. "
                    "NIST tests include frequency, runs, entropy analysis."
                ),
                "example": (
                    "import numpy as np\n"
                    "data = np.array(game_results)\n"
                    "# Frequency test:\n"
                    "median = np.median(data)\n"
                    "above = (data > median).sum()\n"
                    "below = (data <= median).sum()\n"
                    "balance = min(above, below) / max(above, below)\n"
                    "print(f'Balance: {balance:.3f} (>0.9 = good)')"
                ),
            },
        }

    def get_education_topics(self) -> List[str]:
        return list(self.education_topics.keys())

    def learn(self, topic: str) -> Optional[dict]:
        """Get educational content on a security topic."""
        return self.education_topics.get(topic)

    def get_all_education(self) -> dict:
        return self.education_topics

    # ==================================================================
    # 8. PROVABLY FAIR VERIFICATION
    # ==================================================================
    def verify_crash_result(self, server_seed: str, client_seed: str,
                            nonce: int, claimed_result: float) -> dict:
        """
        Verify a provably fair crash game result.
        Uses the standard HMAC-SHA256 algorithm.
        """
        message = f"{client_seed}:{nonce}"
        sig = hmac.new(
            server_seed.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        h = int(sig[:8], 16)
        calculated = max(1.00, (2 ** 32) / (h + 1) * 0.99)
        calculated = round(calculated, 2)

        return {
            "server_seed": server_seed,
            "client_seed": client_seed,
            "nonce": nonce,
            "calculated_result": calculated,
            "claimed_result": claimed_result,
            "match": abs(calculated - claimed_result) < 0.01,
            "hmac_signature": sig[:16] + "...",
        }


# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PYTHON SECURITY ENGINE — Educational Security Tools")
    print("=" * 60)

    engine = SecurityEngine()

    # Demo: Password analysis
    print("\n--- Password Analysis ---")
    for pwd in ["password123", "Str0ng!Pass#2024", "a"]:
        r = engine.analyze_password(pwd)
        print(f"  {pwd:20s} -> {r['rating']} (entropy: {r['entropy_bits']} bits)")

    # Demo: Hash
    print("\n--- Hashing ---")
    print(f"  SHA-256('hello') = {engine.hash_data('hello')[:32]}...")

    # Demo: Port scan (localhost only)
    print("\n--- Port Scan (localhost) ---")
    open_ports = engine.scan_ports("127.0.0.1", [80, 443, 3306, 5432, 8080])
    if open_ports:
        for port, info in open_ports.items():
            print(f"  Port {port}: {info}")
    else:
        print("  No common ports open on localhost")

    # Demo: Code audit
    print("\n--- Code Audit Demo ---")
    test_code = '''
password = "admin123"
os.system("rm -rf " + user_input)
data = pickle.loads(untrusted_data)
    '''
    findings = engine.audit_code_string(test_code)
    for f in findings:
        print(f"  [{f['severity']}] Line {f['line']}: {f['message']}")

    # Demo: Provably fair verification
    print("\n--- Provably Fair Verification ---")
    result = engine.verify_crash_result(
        server_seed="abc123secret",
        client_seed="player_seed_42",
        nonce=1,
        claimed_result=2.45,
    )
    print(f"  Calculated: {result['calculated_result']}")
    print(f"  Match: {result['match']}")

    # Demo: Education
    print("\n--- Security Education Topics ---")
    for topic in engine.get_education_topics():
        info = engine.learn(topic)
        print(f"  * {info['title']}")
