<h1 style="text-align: center"><em>LLM Legal Assistant</em> - A assistant that gives opinion on the questions that falls under EU law</h2>

<h2 style="font-align: center">Natural Language Processing using Transformers</h2>

### Requirements

#### Hardware requirements

#### Software requirements

- python >= 3.11
- pip >= 21.3

### Installation

#### Clone the repository

```bash
git clone https://github.com/mmshress/INLP-WS23.git
cd INLP-WS23
```

#### Check python version

```bash
python --version
```

- if not 3.11 then install python 3.11, from [here](https://www.python.org/downloads/release/python-3110/)
- once installed

#### Note python3.11 path

```bash
# on windows
where python3.11
# on linux
which python3.11
```

#### Setup virtual environment

- Now, use that path to create a virtual environment

```bash
/path/to/python3.11 -m venv .llmcoder.env
# on windows,
## in powershell
.llmcoder.env/Scripts/activate
## in cmd
.llmcoder.env/Scripts/activate.bat
# on linux
source .llmcoder.env/bin/activate
```

#### Setup package

```bash
 python -m pip install .
```

#### For development

```bash
 python -m pip install -e .[dev]
 pre-commit install
```

### Usage

#### Using the CLI

```bash
llmlegalassistant [commands] [options]
```

##### Help

```bash
llmlegalassistant --help
```

# Citation

```bibtex
@software{llmlegalassistant-hd-24,
    author = {Asma Motmem and Siddthant Tripathi and Kushal Gaywala and Mohit Shrestha},
    title = {LLMLegalAssistant: A question answering for EU law},
    month = mar,
    year = 2024,
    publisher = {GitHub},
    version = {0.1},
    url = {https://github.com/mmshress/INLP-WS23}
}
```
