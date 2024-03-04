<h1 align="center" style="margin-top: 0px;"><b>LLM Legal Assistant</b> - A assistant that gives opinion on the questions that falls under EU law</h2>

<h2 align="center" style="margin-top: 0px;">Natural Language Processing using Transformers <br>
 asmaM1 - Asma Motmem 3769398
 <br> KushalGaywala - Kushal Gaywala 3769104
<br> mmshress - Mohit Shrestha 3769398
 <br> sid-code14 - Siddhant Tripathi 3768501
</h2>

#### Software requirements

- python >= 3.11
- pip >= 21.3

### Installation

#### Clone the repository

```bash
git clone https://github.com/mmshress/INLP-WS23.git
cd INLP-WS23
```

#### Clone the data

##### Eval dataset

Unprocessed data and relevant data

```bash
git clone https://huggingface.co/datasets/LLMLegalAssistant/datasets
```

##### Final dataset

Final 511 corpus Indexed, with evaluation small (20 documents) corpus Indexed.

```bash
https://heibox.uni-heidelberg.de/d/91fd4895a03c436f9507/
```

#### Check python version

```bash
python --version
```

- if not 3.11 then install python 3.11, from [here](https://www.python.org/downloads/release/python-3110/)
- once installed

#### Setup virtual environment

- Now, use that path to create a virtual environment

```bash
/path/to/python3.11 -m venv .env
# on windows,
## in powershell
.env/Scripts/Activate.ps1
## in cmd
.env/Scripts/activate.bat
# on linux
source .env/bin/activate
```

#### Setup package

```bash
python -m pip install .
```

#### For development

```bash
 python -m pip install -e '.[dev]'
 pre-commit install
```

### Usage

#### Using the CLI

```bash
llmlegalassistant answer -q <query>
or
llmlegalassistant answer --query <query>
```

- If you want to use OpenAI model for inference then provide a key with `--openapi` flag
- If you want to use LLaMA from HuggingFace then provide your key with `--huggingface` flag
 - we use `meta-llama/Llama-2-7b-chat-hf` which is Quantized with 4-bit Medium, `Q4_K_M`

```bash
llmlegalassistant [commands] [options]
```

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
