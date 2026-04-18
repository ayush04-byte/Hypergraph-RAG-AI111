# Hypergraph-RAG-AI111

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Project-Ongoing-brightgreen)
![Course](https://img.shields.io/badge/Course-AI111-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-blueviolet)

<p align="center">
  <h1> Hypergraph-Based Retrieval System</h1>
</p>

## Overview
This project implements a "Hypergraph-Based Retrieval-Augmented System" for multi-entity data analysis. Traditional graph models are limited to pairwise relationships, whereas hypergraphs allow a single edge (hyperedge) to connect multiple entities simultaneously. 

The system leverages this capability to perform **multi-hop retrieval** and capture complex relationships between queries, documents, and entities. 

## Objectives
- Model multi-entity relationships using hypergraphs
- Enable multi-hop traversal across interconnected data
- Perform retrieval based on connectivity and co-occurrence
- Demonstrate advantages over traditional graph-based approaches

## Key Concepts 
- **Hypergraph**: A generalization of graphs where one edge can connect multiple nodes
- **Hyperedge**: A relationship involving more than two entities
- **Multi-hop Retrieval**: Traversing through multiple connected nodes to gather relevant information
- **Higher-order Relationships**: Capturing complex dependencies beyond pairwise interactions

## Why Hypergraph?

Traditional graphs only support pairwise relationships. However, real-world data often involves multiple entities interacting simultaneously.

Hypergraphs allow:
- Representation of higher-order relationships  
- Better context preservation  
- More efficient multi-hop reasoning  

This makes them suitable for complex retrieval systems.
## Tech Stack
- Python

## Project Structure
Hypergraph-RAG-AI111/<br>
│── src/<br>
│ ├── index & retrieval.py # Hypermem Index<br>
│ ├── Ingestion.py # Build Memory<br>
│ └── Memory Schema_Hypergraph.py # TopicNode, EpisodeNode, FactNode, HypergraphMemory<br>
│ └── Neural Engine.py # HypergraphConv<br>
│ └── demo.py # Demo function<br>
│ └── main.py # Main entry<br>
│<br>
│── results/<br>
│ └── output_examples.txt # Sample outputs<br>
│<br>
│── requirements.txt # Dependencies<br>
gitignore # Ignored files<br>
LICENSE # License File<br>
README.md # README File<br>
---

## Installation & Setup

### 1. Clone the Repository
git clone https://github.com/ayush04-byte/Hypergraph-RAG-AI111.git
cd Hypergraph-RAG-AI111

### 2. Install Dependencies 
pip install -r requirements.txt

### 3. Run the project 
python src/main.py

---

## Example 

![Output](Hypergraph%20RAG%20AI111/example.jpeg)

--- 

## DEMO

![Demo](Hypergraph%20RAG%20AI111/code%20gif.gif)

---

## Features
- Hypergraph contruction using multi-entity relationships
- Multi-hop traversal for deeper data exploration
- Basic retrieval mechanism based on connectivity
- Scalable design for future integration with advanced models

## Future work
- Interpretation with Hypergraph Neural Networks (HGNN)
- Incorporation of Large Language Models (LLMs)
- Use of real-world large-scale datasets
- Advanced ranking and similarity measures

## Limitations
- Prototype-level implementation
- No deep learning integration (yet)
- Limited dataset for demonstration purposes

## Team Members & Roles 
- **Literature Survey** : Nikhil Harsh (2025AIB1041)
- **Coding / Implementation** : Abhigyan Jha (2025AIB1002) , Rijul Bhutani (2025AIB1053)
- **Report / PPT** : Danish Chopra (2025AIB1018)
- **Github Repository** : Ayush Kumar (2025AIB1013)

## Repository Activity

This repository was actively developed with structured commits reflecting incremental progress in:
- Hypergraph construction  
- Retrieval logic  
- Documentation 

## Contribution 

This repository was structured, documented, and maintained to ensure: 
-Clarity of implementation
-Reproducibility of results
-Ease of understanding for evaluation

## Conclusion

This project demonstrates how hypergraphs can effectively model multi-entity relationships and improve retrieval systems through strucuted representation and multi-hop reasoning.

## License 

The MIT License (MIT)

Copyright (c) 2026 ayush04-byte

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
