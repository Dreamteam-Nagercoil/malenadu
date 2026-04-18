const axios = require("axios");
const { createClient } = require("@supabase/supabase-js");

// --- CONFIG ---
const SUPABASE_URL = "http://localhost:54321";
const SUPABASE_KEY = "";
const TELEMETRY_BASE_URL = "http://localhost:3000/stream";
const MODEL_STREAM_URL = "http://localhost:5000/stream";

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
const machineIds = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"];

// --- GLOBAL QUEUE LOGIC ---
let isProcessing = false;
const dataQueue = [];

/**
 * Merged Process Queue: Handles routing to multiple tables
 */
async function processQueue() {
  if (isProcessing || dataQueue.length === 0) return;
  isProcessing = true;

  while (dataQueue.length > 0) {
    const item = dataQueue.shift();

    // Determine target table and clean up the object
    const targetTable = item.target_table || "machine_telemetry";
    delete item.target_table;

    try {
      const { error } = await supabase.from(targetTable).insert([item]);

      if (error) {
        console.error(
          `[DB Error] ${targetTable} | ${item.machine_id}:`,
          error.message,
        );
      } else {
        // Distinct logging for Telemetry vs AI Model
        if (targetTable === "machine_telemetry") {
          console.log(`[DATA] ${item.machine_id} -> Synced (${item.status})`);
        } else {
          console.log(
            `[AI_MODEL] ${item.machine_id} -> Risk: ${(item.risk_score * 100).toFixed(1)}% | Pred: ${item.prediction}`,
          );
        }
      }
    } catch (err) {
      console.error(
        `[CRITICAL] Database unreachable during ${targetTable} sync`,
        err.message,
      );
    }
  }

  isProcessing = false;
}

// --- STREAM 1: TELEMETRY BRIDGE ---
async function startMachineStream(id) {
  console.log(`[SYSTEM] Starting Telemetry Stream for: ${id}`);
  try {
    const response = await axios({
      method: "get",
      url: `${TELEMETRY_BASE_URL}/${id}`,
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
            const data = JSON.parse(line.replace("data: ", "").trim());

            dataQueue.push({
              target_table: "machine_telemetry",
              machine_id: data.machine_id,
              recorded_at: data.timestamp,
              temperature_c: data.temperature_C,
              vibration_mm_s: data.vibration_mm_s,
              rpm: data.rpm,
              current_a: data.current_A,
              status: data.status,
            });

            processQueue();
          } catch (e) {
            /* Ignore heartbeat/empty lines */
          }
        }
      }
    }
  } catch (err) {
    console.error(`[TELEMETRY_FAIL] ${id}: ${err.message}. Retrying...`);
    setTimeout(() => startMachineStream(id), 5000);
  }
}

// --- STREAM 2: AI MODEL BRIDGE ---
async function startModelStream() {
  console.log(`[SYSTEM] Initializing AI Model Prediction Stream...`);
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

            // Iterate through machine keys in the model output
            for (const mId in modelOutput) {
              const prediction = modelOutput[mId];

              dataQueue.push({
                target_table: "model_predictions",
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
            console.error("[MODEL] Parse Error - Check stream format");
          }
        }
      }
    }
  } catch (err) {
    console.error(`[MODEL_FAIL] ${err.message}. Retrying in 10s...`);
    setTimeout(startModelStream, 10000);
  }
}

// --- EXECUTION ---
console.log("-----------------------------------------");
console.log("   INDUSTRIAL AI BRIDGE SERVER ACTIVE    ");
console.log("-----------------------------------------");

// Start all telemetry streams
machineIds.forEach((id) => startMachineStream(id));

// Start the centralized model stream
startModelStream();
