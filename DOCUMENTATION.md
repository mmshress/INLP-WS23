<h1>LLM Legal Assistant</h1>
<h3>
Team Members:
</h3>
Assignment Group 17
<ul>
<li>Siddhant Tripathi (siddhant.tripathi@stud.uni-heidelberg.de)</li>
<li>Asma Motmem (asma.motmem@stud.uni-heidelberg.de)</li>
<li>Kushal Gaywala (kushal.gaywala@stud.uni-heidelberg.de)</li>
<li>Mohit Shrestha (mohit.shrestha@stud.uni-heidelberg.de)</li>
</ul>
<h4>Advisor: Prof. Dr. Michael Gertz</h4>
We hereby certify that we have written the work ourselves and that we have not used any sources or aids other than those specified and that we have marked what has been taken over from other people's works, either verbatim or in terms of content, as foreign.

<h2>Introduction</h2>
In this project, we explore a possible approach to create a question answering system for a domain-specific corpus. In our case, the corpus we based our system on is from the legal domain, specifically all the [EUR-Lex](#EURLex) English language articles from the "energy" domain.  
While large language models (LLMs) have recently become the go-to technology for most chatbot and question answering systems, they come with their own set of limitations. A particular problem that could discourage the use of LLMs for our purpose is the fact that LLMs tend to be trained on a generalized language corpus. This leads to difficulties in receiving answers with information relevant to the specific domain we want answers from. 
To circumvent this, we employ an approach that has gained a lot of traction in recent times: a retrieval-augmented generation (RAG) pipeline.
By retrieving relevant information from the domain-specific corpus before using the LLM to answer questions, we ensure that the answers we receive all come from the corpus, instead of from the general parameters of the LLM. To determine the best possible pipeline for our system, we evaluate multiple embedding models, chunking strategies as well as query retrieval strategies. Finally, we package the optimal pipeline from our evaluations in Python for use as a QA system.       

<h2>Related Work</h2>
[Gao2023](#Gao2023) highlights the limitations of Large Language Models (LLMs), such as hallucination, outdated knowledge, and non-transparent reasoning, 
and introduces Retrieval-Augmented Generation (RAG) as a promising solution. RAG addresses these issues by incorporating knowledge from external databases, 
enhancing accuracy and credibility, particularly for knowledge-intensive tasks. The synergy between RAG and LLMs involves merging intrinsic knowledge with 
dynamic external repositories. Overall, RAG offers a comprehensive solution that combines the strengths of LLMs with external knowledge to overcome their limitations. 
Our project uses RAG in a similar vein to bypass the limitations of LLMs. 

The focus of [Lala2023](#Lala2023)is on applying RAG models, specifically demonstrated by PaperQA, to process scientific knowledge systematically. PaperQA, as a question-answering agent,
conducts information retrieval across full-text scientific articles, evaluates source and passage relevance, and utilizes RAG for answering questions. 
The performance of PaperQA surpasses existing LLMs and LLM agents on current science question-answering benchmarks. Similar to their work on the scientific domain, 
our project aims to compare the relative performance of QA systems on the legal domain. 

[Siriwardhan2023](#Siriwardhana2023) also aims to improve the adaptation of RAG systems to a range of domains, and we use similar techniques in our system for the legal domain. 
Overall, there are almost no standalone papers exploring the performance of RAG systems particularly on the legal domain, and our project aims to provide such a baseline. 

<h2>References</h2>
<ul>
<li id="EURLex">European Union. (n.d.). EUR-Lex portal, access to European law. https://eur-lex.europa.eu/homepage.html</li>
<li id="Gao2023">Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & Wang, H. (2023). Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997.</li>
<li id="Siriwardhana2023">Siriwardhana, S., Weerasekera, R., Wen, E., Kaluarachchi, T., Rana, R., & Nanayakkara, S. (2023). Improving the domain adaptation of retrieval augmented generation (RAG) models for open domain question answering. Transactions of the Association for Computational Linguistics, 11, 1-17.</li>
<li id="Lala2023">Lála, J., O'Donoghue, O., Shtedritski, A., Cox, S., Rodriques, S. G., & White, A. D. (2023). Paperqa: Retrieval-augmented generative agent for scientific research. arXiv preprint arXiv:2312.07559.</li>
</ul>
    