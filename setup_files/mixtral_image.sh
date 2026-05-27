sudo docker pull ollama/ollama
sudo docker run -d --gpus=all -v ollama_mixtral:/root/.ollama -p 11435:11434 --name ollama_mixtral ollama/ollama
sudo docker exec -it ollama_mixtral ollama pull mixtral:8x7b
sudo docker exec -it ollama_mixtral ollama run mixtral:8x7b
