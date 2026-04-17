const MODEL_STREAM_URL = "http://100.107.213.66:5000/stream";

async function startModelStream() {
  console.log(`[SYSTEM] Initializing AI Model Stream...`);
  try {
    const response = await axios({
      method: "get",
      url: MODEL_STREAM_URL,
      responseType: "stream",
      timeout: 0,
    });

    let buffer = "";

    for await (const chunk of response.data) {
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.trim().startsWith("data: ")) {
          try {
            const rawJson = line.replace("data: ", "").trim();
            const modelOutput = JSON.parse(rawJson);

            // The model sends an object where keys are machine IDs
            // e.g., {"CNC_01": {...}, "CNC_02": {...}}
            for (const mId in modelOutput) {
              const prediction = modelOutput[mId];

              dataQueue.push({
                table: "model_predictions", // Custom flag for our queue
                machine_id: prediction.machine_id,
                risk_score: prediction.risk,
                prediction: prediction.prediction,
                actual_status: prediction.actual,
                accuracy: prediction.accuracy,
                is_learning: prediction.learning,
              });
            }
            processQueue();
          } catch (e) {
            console.error("Model Parse Error");
          }
        }
      }
    }
  } catch (err) {
    console.error(`[MODEL_FAIL] ${err.message}. Retrying...`);
    setTimeout(startModelStream, 10000);
  }
}

// Update your processQueue to handle different tables
async function processQueue() {
  if (isProcessing || dataQueue.length === 0) return;
  isProcessing = true;

  while (dataQueue.length > 0) {
    const item = dataQueue.shift();
    const targetTable = item.table || "machine_telemetry";
    delete item.table; // Clean up before insert

    try {
      await supabase.from(targetTable).insert([item]);
    } catch (err) {
      console.error("DB Insert Error");
    }
  }
  isProcessing = false;
}

// Start the model stream along with machine streams
startModelStream();
