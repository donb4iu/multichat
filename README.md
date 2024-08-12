# multichat
Langchain &amp; Streamlit


## References

[Building a Multi PDF RAG Chatbot: Langchain, Streamlit with code](https://medium.com/gopenai/building-a-multi-pdf-rag-chatbot-langchain-streamlit-with-code-d21d0a1cf9e5)

## Requirements

### #( 08/12/24@ 6:26PM )( donbuddenbaum@donbs-imac ):~
   pip install PyPDF2

```
Collecting PyPDF2
Installing collected packages: PyPDF2
Successfully installed PyPDF2-3.0.1
```

### #( 08/12/24@ 6:32PM )( donbuddenbaum@donbs-imac ):~/Documents/multichat/src@main✗✗✗
   pip install langchain-anthropic

```
Collecting langchain-anthropic
  Downloading langchain_anthropic-0.1.22-py3-none-any.whl.metadata (2.2 kB)
  Successfully installed anthropic-0.33.1 distro-1.9.0 jiter-0.5.0 langchain-anthropic-0.1.22
  ```

### #( 08/12/24@ 6:34PM )( donbuddenbaum@donbs-imac ):~/Documents/multichat/src@main✗✗✗
   pip install langchain_openai

```
Collecting langchain_openai
Installing collected packages: tiktoken, openai, langchain_openai
Successfully installed langchain_openai-0.1.21 openai-1.40.6 tiktoken-0.7.0
```

### #( 08/12/24@ 6:38PM )( donbuddenbaum@donbs-imac ):~/Documents/multichat/src@main✗✗✗
   pip install spacy

```
Collecting spacy
Installing collected packages: cymem, wasabi, spacy-loggers, spacy-legacy, smart-open, murmurhash, marisa-trie, cloudpathlib, catalogue, blis, srsly, preshed, language-data, langcodes, confection, weasel, thinc, spacy
Successfully installed blis-0.7.11 catalogue-2.0.10 cloudpathlib-0.18.1 confection-0.1.5 cymem-2.0.8 langcodes-3.4.0 language-data-1.2.0 marisa-trie-1.2.0 murmurhash-1.0.10 preshed-3.0.9 smart-open-7.0.4 spacy-3.7.5 spacy-legacy-3.0.12 spacy-loggers-1.0.5 srsly-2.4.8 thinc-8.2.5 wasabi-1.1.3 weasel-0.4.1
```

### #( 08/12/24@ 6:44PM )( donbuddenbaum@donbs-imac ):~/Documents/multichat/src@main✗✗✗
   python -m spacy download en_core_web_sm

```
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-3.7.1
```

### #( 08/12/24@ 6:57PM )( donbuddenbaum@donbs-imac ):~/Documents/multichat@main✗✗✗
   pip install -r requirements.txt

```
Collecting faiss-cpu (from -r requirements.txt (line 9))
Installing collected packages: faiss-cpu
Successfully installed faiss-cpu-1.8.0.post1
```