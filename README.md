# Instruction to setup

Install docker on your system based on your operating system and install SupaBase.

# Install Dependencies

```bash
npm install
```

After all the dependencies have setup (including pip). Run the servers in the following order

```bash
node bridge.js
python agent_db.py
python dashboard_server.py
python nlp2.py
```

# NLP Setup
For the NLP, setup Ollama and install "llama3.2:3b". Make sure the Ollama server is running.

> Note: You have to enter your Supabase private key in the files for the database to connect.
