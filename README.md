### GenXAI (Explainable/Interpretanble GenAI)

#### GenAI Interpretability is a powerful concept in the field of artificial intelligence that aims to provide insights and understanding into the decision-making process of the models. It focuses on explaining how and why a model arrives at a particular prediction or decision, making it easier for humans to interpret and trust the model's outputs.

I surveyed the list of known techniques available so far and captured them in the following mindmap.

![Explainable GenAI Techniques](https://github.com/spraja08/Interpretable-GenAI/blob/main/resources/GenXAI%20Methods.png)

Refer to the genai_interp.ipynb for a demo of selected techniques as below:

1. Mehcanistic (Head, Model Visualisation)
2. Feature Attribution (Integrated Gradients-based)
3. Sample-based (Counterfactuals-based)

To install ollama locally (docker based):
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama --restart always ollama/ollama
docker exec -it ollama ollama pull llama3
docker exec -it ollama ollama pull mxbai-embed-large

