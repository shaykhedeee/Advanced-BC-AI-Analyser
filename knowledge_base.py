"""
knowledge_base.py — Expandable Knowledge Base
Stores Kali Linux tools, algorithms, security techniques in compressed memory-efficient format.
Auto-loads into AI Brain for continuous learning without massive storage usage.

Knowledge categories:
  - Kali Linux tools (network reconnaissance, scanning, exploitation)
  - Algorithms (sorting, searching, graphs, DP, crypto)
  - Python security patterns
  - Game theory & probability
  - Data structures
"""

import json
import zlib
import base64
from typing import Dict, List, Any, Optional


class KnowledgeBase:
    """Memory-efficient knowledge base with compression."""

    def __init__(self, compress: bool = True):
        self.compress = compress
        self._kb = {}
        self._metadata = {"total_entries": 0, "categories": []}
        self._build_knowledge_base()

    def _build_knowledge_base(self):
        """Build all knowledge categories."""
        self.add_kali_linux_knowledge()
        self.add_algorithms_knowledge()
        self.add_python_patterns_knowledge()
        self.add_game_theory_knowledge()
        self.add_security_techniques()

    # ================================================================
    # KALI LINUX KNOWLEDGE
    # ================================================================
    def add_kali_linux_knowledge(self):
        """Kali Linux tools and penetration testing techniques."""
        kali_tools = {
            "category": "Kali Linux Tools",
            "tools": [
                {
                    "name": "nmap",
                    "description": "Network mapper - scans networks for open ports and services",
                    "use_cases": ["host discovery", "port scanning", "service detection", "OS detection"],
                    "python_equivalent": (
                        "import socket, threading\n"
                        "def scan_port(host, port, timeout=1):\n"
                        "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                        "    s.settimeout(timeout)\n"
                        "    result = s.connect_ex((host, port))\n"
                        "    return result == 0"
                    ),
                },
                {
                    "name": "metasploit",
                    "description": "Exploitation framework with payloads and modules",
                    "use_cases": ["vulnerability exploitation", "payload generation", "post-exploitation"],
                    "python_equivalent": (
                        "# Python-based framework concept\n"
                        "class Exploit:\n"
                        "    def __init__(self, target, payload):\n"
                        "        self.target = target\n"
                        "        self.payload = payload\n"
                        "    def deliver(self):\n"
                        "        ...  # Send payload\n"
                        "    def execute(self):\n"
                        "        ...  # Execute on target"
                    ),
                },
                {
                    "name": "wireshark",
                    "description": "Packet sniffer and analyzer for network traffic",
                    "use_cases": ["traffic analysis", "protocol debugging", "intrusion detection"],
                    "python_equivalent": (
                        "# Using scapy\n"
                        "from scapy.all import sniff, IP\n"
                        "def packet_callback(packet):\n"
                        "    if IP in packet:\n"
                        "        print(f'{packet[IP].src} -> {packet[IP].dst}')\n"
                        "sniff(prn=packet_callback, count=10)"
                    ),
                },
                {
                    "name": "john_the_ripper",
                    "description": "Password cracker for hash/password cracking",
                    "use_cases": ["password auditing", "hash cracking", "brute force"],
                    "python_equivalent": (
                        "import hashlib, itertools\n"
                        "def crack_hash(target_hash, wordlist):\n"
                        "    for word in wordlist:\n"
                        "        h = hashlib.sha256(word.encode()).hexdigest()\n"
                        "        if h == target_hash:\n"
                        "            return word\n"
                        "    return None"
                    ),
                },
                {
                    "name": "sqlmap",
                    "description": "SQL injection detection and exploitation",
                    "use_cases": ["SQL injection testing", "database extraction", "privilege escalation"],
                    "python_equivalent": (
                        "import requests\n"
                        "def test_sql_injection(url, param):\n"
                        "    payloads = [\n"
                        "        \"' OR '1'='1\",\n"
                        "        \"' UNION SELECT NULL--\"\n"
                        "    ]\n"
                        "    for payload in payloads:\n"
                        "        r = requests.get(url, params={param: payload})\n"
                        "        if 'error' in r.text:\n"
                        "            return True\n"
                        "    return False"
                    ),
                },
                {
                    "name": "hashcat",
                    "description": "GPU-accelerated password recovery",
                    "use_cases": ["GPU cracking", "rainbow tables", "rule-based attacks"],
                    "python_equivalent": (
                        "# Using GPU libraries like CuPy for acceleration\n"
                        "def gpu_hash_generation(charset, max_len):\n"
                        "    # Generate hashes on GPU\n"
                        "    # Compare with target\n"
                        "    pass"
                    ),
                },
                {
                    "name": "aircrack-ng",
                    "description": "Wi-Fi security auditing and WEP/WPA cracking",
                    "use_cases": ["wireless testing", "packet capture", "key recovery"],
                    "python_equivalent": (
                        "# Using scapy for wireless\n"
                        "from scapy.all import ARP, Ether, srp\n"
                        "def scan_wifi_networks():\n"
                        "    arp_request = Ether(dst='ff:ff:ff:ff:ff:ff')/ARP(pdst='192.168.1.0/24')\n"
                        "    answered, _ = srp(arp_request, timeout=2, verbose=False)\n"
                        "    return answered"
                    ),
                },
                {
                    "name": "burpsuite",
                    "description": "Web application security testing platform",
                    "use_cases": ["intercepting proxies", "web scanning", "vulnerability assessment"],
                    "python_equivalent": (
                        "import requests\n"
                        "def web_test(url):\n"
                        "    # Intercept, modify, replay\n"
                        "    r = requests.get(url)\n"
                        "    # Test for XSS, CSRF, SQLi\n"
                        "    pass"
                    ),
                },
                {
                    "name": "theHarvester",
                    "description": "OSINT tool for gathering email addresses and subdomains",
                    "use_cases": ["reconnaissance", "OSINT", "domain mapping"],
                    "python_equivalent": (
                        "import requests, re\n"
                        "def harvest_emails(domain):\n"
                        "    emails = []\n"
                        "    # Query search engines\n"
                        "    # Extract email patterns\n"
                        "    return emails"
                    ),
                },
            ]
        }
        self._kb["kali_linux"] = kali_tools
        self._metadata["categories"].append("kali_linux")

    # ================================================================
    # ALGORITHMS KNOWLEDGE
    # ================================================================
    def add_algorithms_knowledge(self):
        """Algorithm implementations and complexity analysis."""
        algorithms = {
            "category": "Algorithms",
            "subcategories": {
                "sorting": [
                    {
                        "name": "Quicksort",
                        "time_complexity": "O(n log n) avg, O(n²) worst",
                        "space_complexity": "O(log n) aux",
                        "use_cases": ["general purpose", "in-place sorting"],
                        "code": (
                            "def quicksort(arr):\n"
                            "    if len(arr) <= 1:\n"
                            "        return arr\n"
                            "    pivot = arr[len(arr)//2]\n"
                            "    left = [x for x in arr if x < pivot]\n"
                            "    mid = [x for x in arr if x == pivot]\n"
                            "    right = [x for x in arr if x > pivot]\n"
                            "    return quicksort(left) + mid + quicksort(right)"
                        ),
                    },
                    {
                        "name": "Mergesort",
                        "time_complexity": "O(n log n) guaranteed",
                        "space_complexity": "O(n) aux",
                        "use_cases": ["stable sorting", "external sorting"],
                        "code": (
                            "def mergesort(arr):\n"
                            "    if len(arr) <= 1:\n"
                            "        return arr\n"
                            "    mid = len(arr) // 2\n"
                            "    left = mergesort(arr[:mid])\n"
                            "    right = mergesort(arr[mid:])\n"
                            "    return merge(left, right)\n"
                            "def merge(a, b):\n"
                            "    result = []\n"
                            "    i = j = 0\n"
                            "    while i < len(a) and j < len(b):\n"
                            "        if a[i] < b[j]:\n"
                            "            result.append(a[i])\n"
                            "            i += 1\n"
                            "        else:\n"
                            "            result.append(b[j])\n"
                            "            j += 1\n"
                            "    return result + a[i:] + b[j:]"
                        ),
                    },
                ],
                "searching": [
                    {
                        "name": "Binary Search",
                        "time_complexity": "O(log n)",
                        "prerequisites": "sorted data",
                        "use_cases": ["finding elements", "insertion point"],
                        "code": (
                            "def binary_search(arr, target):\n"
                            "    left, right = 0, len(arr) - 1\n"
                            "    while left <= right:\n"
                            "        mid = (left + right) // 2\n"
                            "        if arr[mid] == target:\n"
                            "            return mid\n"
                            "        elif arr[mid] < target:\n"
                            "            left = mid + 1\n"
                            "        else:\n"
                            "            right = mid - 1\n"
                            "    return -1"
                        ),
                    },
                ],
                "graphs": [
                    {
                        "name": "Dijkstra's Algorithm",
                        "type": "shortest path",
                        "time_complexity": "O((V+E) log V)",
                        "use_cases": ["GPS navigation", "network routing"],
                        "code": (
                            "import heapq\n"
                            "def dijkstra(graph, start):\n"
                            "    distances = {node: float('inf') for node in graph}\n"
                            "    distances[start] = 0\n"
                            "    pq = [(0, start)]\n"
                            "    while pq:\n"
                            "        current_dist, current = heapq.heappop(pq)\n"
                            "        if current_dist > distances[current]:\n"
                            "            continue\n"
                            "        for neighbor, weight in graph[current]:\n"
                            "            distance = current_dist + weight\n"
                            "            if distance < distances[neighbor]:\n"
                            "                distances[neighbor] = distance\n"
                            "                heapq.heappush(pq, (distance, neighbor))\n"
                            "    return distances"
                        ),
                    },
                    {
                        "name": "BFS (Breadth-First Search)",
                        "type": "graph traversal",
                        "time_complexity": "O(V + E)",
                        "use_cases": ["level-order traversal", "shortest path in unweighted"],
                        "code": (
                            "from collections import deque\n"
                            "def bfs(graph, start):\n"
                            "    visited = set()\n"
                            "    queue = deque([start])\n"
                            "    visited.add(start)\n"
                            "    while queue:\n"
                            "        node = queue.popleft()\n"
                            "        print(node)\n"
                            "        for neighbor in graph[node]:\n"
                            "            if neighbor not in visited:\n"
                            "                visited.add(neighbor)\n"
                            "                queue.append(neighbor)"
                        ),
                    },
                ],
                "dynamic_programming": [
                    {
                        "name": "Fibonacci (Memoization)",
                        "time_complexity": "O(n)",
                        "space_complexity": "O(n)",
                        "use_cases": ["sequence generation", "optimization"],
                        "code": (
                            "def fibonacci(n, memo={}):\n"
                            "    if n in memo:\n"
                            "        return memo[n]\n"
                            "    if n <= 1:\n"
                            "        return n\n"
                            "    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)\n"
                            "    return memo[n]"
                        ),
                    },
                    {
                        "name": "Knapsack Problem",
                        "type": "optimization",
                        "time_complexity": "O(n*W) where W is capacity",
                        "use_cases": ["resource allocation", "optimal packing"],
                        "code": (
                            "def knapsack(capacity, weights, values, n):\n"
                            "    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n"
                            "    for i in range(1, n + 1):\n"
                            "        for w in range(1, capacity + 1):\n"
                            "            if weights[i-1] <= w:\n"
                            "                dp[i][w] = max(\n"
                            "                    values[i-1] + dp[i-1][w-weights[i-1]],\n"
                            "                    dp[i-1][w]\n"
                            "                )\n"
                            "            else:\n"
                            "                dp[i][w] = dp[i-1][w]\n"
                            "    return dp[n][capacity]"
                        ),
                    },
                ],
                "cryptography": [
                    {
                        "name": "SHA-256 Hashing",
                        "type": "hash function",
                        "properties": ["deterministic", "one-way", "avalanche effect"],
                        "use_cases": ["data integrity", "password hashing", "blockchain"],
                        "code": (
                            "import hashlib\n"
                            "def sha256_hash(data):\n"
                            "    h = hashlib.sha256()\n"
                            "    h.update(data.encode() if isinstance(data, str) else data)\n"
                            "    return h.hexdigest()\n"
                            "def verify_hash(data, expected_hash):\n"
                            "    return sha256_hash(data) == expected_hash"
                        ),
                    },
                    {
                        "name": "RSA Encryption",
                        "type": "asymmetric cryptography",
                        "properties": ["public-private key pair", "slow", "large key sizes"],
                        "use_cases": ["digital signatures", "key exchange", "secure communication"],
                        "code": (
                            "from cryptography.hazmat.primitives.asymmetric import rsa\n"
                            "from cryptography.hazmat.primitives import hashes\n"
                            "def generate_keypair():\n"
                            "    return rsa.generate_private_key(\n"
                            "        public_exponent=65537,\n"
                            "        key_size=2048\n"
                            "    )"
                        ),
                    },
                ],
                "machine_learning": [
                    {
                        "name": "K-Means Clustering",
                        "type": "unsupervised learning",
                        "time_complexity": "O(n*k*i*d) where i=iterations",
                        "use_cases": ["customer segmentation", "image compression"],
                        "code": (
                            "import numpy as np\n"
                            "def kmeans(X, k, max_iters=100):\n"
                            "    centroids = X[np.random.choice(X.shape[0], k)]\n"
                            "    for _ in range(max_iters):\n"
                            "        distances = np.linalg.norm(X[:, None] - centroids, axis=2)\n"
                            "        labels = np.argmin(distances, axis=1)\n"
                            "        new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(k)])\n"
                            "        if np.allclose(centroids, new_centroids):\n"
                            "            break\n"
                            "        centroids = new_centroids\n"
                            "    return labels, centroids"
                        ),
                    },
                ],
            }
        }
        self._kb["algorithms"] = algorithms
        self._metadata["categories"].append("algorithms")

    # ================================================================
    # PYTHON PATTERNS & SECURITY
    # ================================================================
    def add_python_patterns_knowledge(self):
        """Advanced Python patterns and best practices."""
        patterns = {
            "category": "Python Patterns",
            "patterns": [
                {
                    "name": "Context Manager Pattern",
                    "use_case": "Resource management (files, connections)",
                    "code": (
                        "class FileManager:\n"
                        "    def __init__(self, filename):\n"
                        "        self.filename = filename\n"
                        "    def __enter__(self):\n"
                        "        self.file = open(self.filename, 'r')\n"
                        "        return self.file\n"
                        "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
                        "        self.file.close()\n"
                        "# Usage: with FileManager('file.txt') as f: ..."
                    ),
                },
                {
                    "name": "Decorator Pattern",
                    "use_case": "Function wrapping, logging, caching",
                    "code": (
                        "def cache_decorator(func):\n"
                        "    cache = {}\n"
                        "    def wrapper(*args):\n"
                        "        if args not in cache:\n"
                        "            cache[args] = func(*args)\n"
                        "        return cache[args]\n"
                        "    return wrapper\n"
                        "@cache_decorator\n"
                        "def expensive_function(x):\n"
                        "    return x ** 2"
                    ),
                },
                {
                    "name": "Factory Pattern",
                    "use_case": "Creating objects based on type",
                    "code": (
                        "class DatabaseFactory:\n"
                        "    @staticmethod\n"
                        "    def create(db_type):\n"
                        "        if db_type == 'mysql':\n"
                        "            return MySQLDatabase()\n"
                        "        elif db_type == 'postgres':\n"
                        "            return PostgresDatabase()\n"
                        "        raise ValueError(f'Unknown type: {db_type}')"
                    ),
                },
            ]
        }
        self._kb["python_patterns"] = patterns
        self._metadata["categories"].append("python_patterns")

    # ================================================================
    # GAME THEORY & PROBABILITY
    # ================================================================
    def add_game_theory_knowledge(self):
        """Game theory, probability, and betting strategies."""
        game_theory = {
            "category": "Game Theory & Probability",
            "concepts": [
                {
                    "name": "Expected Value",
                    "formula": "E(X) = Σ(probability * outcome)",
                    "use_cases": ["decision making", "betting strategies"],
                    "example": (
                        "# Coin flip: heads = +$1, tails = -$1\n"
                        "ev = 0.5 * 1 + 0.5 * (-1) = 0\n"
                        "# Fair game (no edge)"
                    ),
                },
                {
                    "name": "Variance & Standard Deviation",
                    "formula": "σ² = E(X²) - E(X)²",
                    "use_cases": ["risk assessment", "bankroll management"],
                    "code": (
                        "import numpy as np\n"
                        "def calculate_variance(outcomes, probabilities):\n"
                        "    expected = sum(o*p for o,p in zip(outcomes, probabilities))\n"
                        "    variance = sum(p*(o-expected)**2 for o,p in zip(outcomes, probabilities))\n"
                        "    return variance, np.sqrt(variance)"
                    ),
                },
                {
                    "name": "Kelly Criterion",
                    "formula": "f* = (bp - q) / b",
                    "use_cases": ["optimal betting", "long-term growth"],
                    "explanation": "b=odds, p=win_prob, q=1-p",
                },
            ]
        }
        self._kb["game_theory"] = game_theory
        self._metadata["categories"].append("game_theory")

    # ================================================================
    # SECURITY TECHNIQUES
    # ================================================================
    def add_security_techniques(self):
        """Advanced security techniques and attack/defense methods."""
        security = {
            "category": "Security Techniques",
            "attack_methods": [
                {
                    "name": "SQL Injection",
                    "attack_vector": "Malicious SQL in user input",
                    "example": "username = \" OR '1'='1' --",
                    "defense": (
                        "# Use parameterized queries\n"
                        "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))\n"
                        "# Never concatenate user input"
                    ),
                },
                {
                    "name": "Cross-Site Scripting (XSS)",
                    "attack_vector": "JavaScript injection in web pages",
                    "example": "comment = \"<script>alert('xss')</script>\"",
                    "defense": (
                        "# HTML escape user input\n"
                        "from html import escape\n"
                        "safe_comment = escape(user_comment)"
                    ),
                },
                {
                    "name": "Man-in-the-Middle (MITM)",
                    "attack_vector": "Intercepting unencrypted traffic",
                    "defense": "Use HTTPS, TLS/SSL, certificate pinning",
                },
            ],
            "defense_strategies": [
                {
                    "name": "Input Validation",
                    "technique": "Verify all user inputs before processing",
                    "types": ["type checking", "range validation", "format validation"],
                },
                {
                    "name": "Authentication",
                    "technique": "Verify user identity",
                    "types": ["passwords", "MFA", "biometric"],
                },
                {
                    "name": "Authorization",
                    "technique": "Control access to resources",
                    "types": ["RBAC", "ABAC", "ACL"],
                },
            ]
        }
        self._kb["security_techniques"] = security
        self._metadata["categories"].append("security_techniques")

    # ================================================================
    # RETRIEVAL & COMPRESSION
    # ================================================================
    def get_knowledge(self, category: str) -> Dict[str, Any]:
        """Get knowledge by category."""
        return self._kb.get(category, {})

    def get_all_knowledge(self) -> Dict[str, Any]:
        """Get all knowledge (for AI training)."""
        return self._kb

    def get_training_examples(self) -> List[Dict[str, str]]:
        """Convert knowledge to training examples for AI model."""
        examples = []
        for category, content in self._kb.items():
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and "code" in item:
                                examples.append({
                                    "instruction": f"Explain and implement: {item.get('name', '')}",
                                    "output": item["code"],
                                    "type": category,
                                })
                            elif isinstance(item, dict) and "output" in item:
                                examples.append({
                                    "instruction": item.get("instruction", item.get("name", "")),
                                    "output": item["output"],
                                    "type": category,
                                })
        return examples

    def compress_knowledge(self) -> str:
        """Compress knowledge base to string for storage."""
        json_str = json.dumps(self._kb)
        compressed = zlib.compress(json_str.encode())
        return base64.b64encode(compressed).decode()

    def decompress_knowledge(self, compressed: str):
        """Decompress knowledge base from string."""
        try:
            decompressed = zlib.decompress(base64.b64decode(compressed))
            self._kb = json.loads(decompressed)
        except Exception as e:
            print(f"Decompression error: {e}")

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about knowledge base."""
        self._metadata["total_entries"] = len(self._kb)
        self._metadata["total_compressed_size"] = len(self.compress_knowledge()) / 1024  # KB
        return self._metadata


# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("KNOWLEDGE BASE — AI Brain Training Data")
    print("=" * 70)

    kb = KnowledgeBase()
    meta = kb.get_metadata()
    
    print(f"\n✅ Knowledge Base Initialized")
    print(f"   Categories: {', '.join(meta['categories'])}")
    print(f"   Total entries: {meta['total_entries']}")
    print(f"   Compressed size: {meta['total_compressed_size']:.2f} KB")

    # Show sample training examples
    examples = kb.get_training_examples()
    print(f"\n📚 Training Examples: {len(examples)}")
    for i, ex in enumerate(examples[:3]):
        print(f"\n  Example {i+1}:")
        print(f"    Instruction: {ex['instruction'][:50]}...")
        print(f"    Type: {ex['type']}")

    # Show Kali tools
    kali = kb.get_knowledge("kali_linux")
    print(f"\n🎯 Kali Linux Tools: {len(kali.get('tools', []))}")
    for tool in kali.get('tools', [])[:3]:
        print(f"   • {tool['name']}: {tool['description'][:50]}...")

    # Show algorithms
    algos = kb.get_knowledge("algorithms")
    print(f"\n🔧 Algorithm Categories: {list(algos.get('subcategories', {}).keys())}")
