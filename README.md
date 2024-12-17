# SDP

## Overview

--- <br>
---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

List the requirements needed to run your project:
- Gem5 23.1.0
- McPat
- Ubuntu 22.04

## Installation

Step-by-step guide to set up the project:

```bash
# Clone the repository
git clone --recursive https://github.com/Jagadeesh-pradhani/SDP.git
git submodule init

# Navigate to the project directory
cd SDP


```

## Project Structure

Explain the directory structure and the purpose of key files and directories:

```
SDP/
│
├── Gem5Mcpat_Parser_2024/  # Parser
│   ├── parser.py
│   └── templates/
│       └──template_latest.xml
│
├── mcpat/              # mcPat source
│   
│
│
├── config/             # Configuration files
│   ├── new/
│   ├── old/
│   └── plots/ 
├── sdp.py              # Source code
│
│
├── requirements.txt    # Python dependencies
├── package.json        # Node.js dependencies
└── README.md           # This file
```

## Usage

Provide clear examples of how to use your project:

```bash
# Basic usage example
python main.py

```



## Contributing

Guidelines for contributing to the project:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request





