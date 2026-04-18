/**
 * Daily Machine Telemetry PDF Report Generator
 *
 * Requirements:
 * npm install @supabase/supabase-js pdfkit chartjs-node-canvas chart.js node-fetch
 *
 * Usage:
 * node generateReport.js [YYYY-MM-DD]
 * node generateReport.js 2025-07-10
 * (defaults to today if no date provided)
 */

const fs   = require("fs");
const path = require("path");
const PDFDocument           = require("pdfkit");
const { createClient }      = require("@supabase/supabase-js");
const { ChartJSNodeCanvas }  = require("chartjs-node-canvas");

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const SUPABASE_URL = "http://100.125.109.107:54321/"; // Make sure this is the API URL, not Studio!
const SUPABASE_KEY = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH";

const MACHINE_IDS = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"];

// Any row whose status (case-insensitive) is in this list is treated as a fault
const FAULT_STATUSES = ["fault", "error", "warning", "critical", "deviated"];

// ─── HELPERS ─────────────────────────────────────────────────────────────────
function getTargetDate() {
  const arg = process.argv[2];
  if (arg && /^\d{4}-\d{2}-\d{2}$/.test(arg)) return arg;
  return new Date().toISOString().slice(0, 10);
}

function isFault(row) {
  return FAULT_STATUSES.includes((row.status ?? "").toLowerCase());
}

/** Simple risk score derived from status string alone */
function riskScore(row) {
  const s = (row.status ?? "").toLowerCase();
  if (s === "critical")  return 90;
  if (s === "fault")     return 75;
  if (s === "error")     return 70;
  if (s === "deviated")  return 55;
  if (s === "warning")   return 40;
  return 10;
}

function fmt(v, digits = 1) {
  return v === null || v === undefined ? "—" : Number(v).toFixed(digits);
}

function fmtTime(ts) {
  return new Date(ts).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

/** Group rows into 4 × 6-hour buckets */
function aggregate6h(rows) {
  const LABELS = ["00:00 – 06:00", "06:00 – 12:00", "12:00 – 18:00", "18:00 – 24:00"];
  const buckets = [[], [], [], []];
  for (const r of rows) {
    const h = new Date(r.recorded_at).getHours();
    buckets[Math.floor(h / 6)].push(r);
  }
  return LABELS.map((label, i) => {
    const g = buckets[i];
    if (!g.length) return { label, count: 0, avg_temp: null, avg_vib: null, avg_rpm: null, avg_cur: null, faults: 0 };
    const avg = (f) => g.reduce((s, r) => s + (r[f] ?? 0), 0) / g.length;
    return {
      label,
      count:    g.length,
      avg_temp: avg("temperature_c"),
      avg_vib:  avg("vibration_mm_s"),
      avg_rpm:  avg("rpm"),
      avg_cur:  avg("current_a"),
      faults:   g.filter(isFault).length,
    };
  });
}

// ─── CHART ───────────────────────────────────────────────────────────────────
async function buildLineChart(rows, field, label, unit, color) {
  const renderer   = new ChartJSNodeCanvas({ width: 480, height: 155, backgroundColour: "white" });
  const sorted     = [...rows].sort((a, b) => new Date(a.recorded_at) - new Date(b.recorded_at));
  const labels     = sorted.map((r) => fmtTime(r.recorded_at));
  const values     = sorted.map((r) => r[field] ?? null);
  const ptColors   = sorted.map((r) => isFault(r) ? "red" : color);

  return renderer.renderToBuffer({
    type: "line",
    data: {
      labels,
      datasets: [{
        label: `${label} (${unit})`,
        data:  values,
        borderColor: color,
        pointBackgroundColor: ptColors,
        pointRadius: ptColors.map((c) => (c === "red" ? 5 : 2)),
        segment: { borderColor: (ctx) => isFault(sorted[ctx.p1DataIndex]) ? "red" : color },
        tension: 0.3,
        fill:    false,
      }],
    },
    options: {
      animation: false,
      plugins: { legend: { labels: { font: { size: 10 } } } },
      scales: {
        x: { ticks: { font: { size: 8 }, maxTicksLimit: 12 } },
        y: { ticks: { font: { size: 9 } } },
      },
    },
  });
}

// ─── PDF HELPERS ─────────────────────────────────────────────────────────────
const MARGIN = 40;
const PAGE_W = 595;
const COL_W  = PAGE_W - MARGIN * 2;

function sectionTitle(doc, text) {
  doc.moveDown(1).font("Helvetica-Bold").fontSize(11).fillColor("#333333").text(text, MARGIN).moveDown(0.3);
}

function drawHRule(doc) {
  doc.moveTo(MARGIN, doc.y).lineTo(PAGE_W - MARGIN, doc.y).strokeColor("#cccccc").lineWidth(0.5).stroke();
  doc.moveDown(0.4);
}

function drawTable(doc, headers, rows, colWidths) {
  const ROW_H = 16;
  
  // Helper function to draw headers (useful for pagination)
  function drawHeader() {
    doc.rect(MARGIN, doc.y, COL_W, ROW_H).fill("#555555");
    const hy = doc.y + 4;
    doc.font("Helvetica-Bold").fontSize(8).fillColor("#ffffff");
    let hx = MARGIN;
    for (let i = 0; i < headers.length; i++) {
      doc.text(headers[i], hx + 3, hy, { width: colWidths[i] - 6, lineBreak: false });
      hx += colWidths[i];
    }
    doc.y += ROW_H;
  }

  // Check if we need a new page before starting the table
  if (doc.y + ROW_H * 2 > doc.page.height - 60) doc.addPage();
  drawHeader();

  // Draw data rows
  rows.forEach((row, ri) => {
    // If running out of space, add a page and reprint the header
    if (doc.y + ROW_H > doc.page.height - 60) {
      doc.addPage();
      drawHeader();
    }
    
    const ry = doc.y;
    doc.rect(MARGIN, ry, COL_W, ROW_H).fill(ri % 2 === 0 ? "#f9f9f9" : "#ffffff");
    let x = MARGIN;
    doc.font("Helvetica").fontSize(7.5).fillColor("#222222");
    for (let i = 0; i < row.length; i++) {
      doc.text(String(row[i]), x + 3, ry + 4, { width: colWidths[i] - 6, lineBreak: false });
      x += colWidths[i];
    }
    doc.y = ry + ROW_H;
  });
  doc.moveDown(0.5);
}

// ─── PDF BUILDER ─────────────────────────────────────────────────────────────
async function buildPDF(date, allRows, outputPath) {
  const doc    = new PDFDocument({ size: "A4", margins: { top: MARGIN, bottom: 50, left: MARGIN, right: MARGIN }, autoFirstPage: true, bufferPages: true });
  const stream = fs.createWriteStream(outputPath);
  doc.pipe(stream);

  // Title page header
  doc.font("Helvetica-Bold").fontSize(16).fillColor("#111111")
     .text("Machine Telemetry — Daily Report", MARGIN, MARGIN, { align: "center" });
  doc.font("Helvetica").fontSize(10).fillColor("#555555")
     .text(`Report Date: ${date}  |  Machines: ${MACHINE_IDS.join(", ")}`, { align: "center" });
  doc.moveDown(0.5);
  drawHRule(doc);

  for (const machineId of MACHINE_IDS) {
    const rows   = allRows.filter((r) => r.machine_id === machineId);
    const faults = rows.filter(isFault);

    // Machine heading
    doc.font("Helvetica-Bold").fontSize(13).fillColor("#1a1a1a").text(machineId, MARGIN);
    doc.font("Helvetica").fontSize(9).fillColor("#666666").text(
      rows.length === 0
        ? "No data recorded for this day."
        : `${rows.length} readings  |  ${fmtTime(rows[0].recorded_at)} – ${fmtTime(rows[rows.length - 1].recorded_at)}  |  Faults: ${faults.length}`,
    );
    doc.moveDown(0.4);

    if (rows.length === 0) { drawHRule(doc); continue; }

    // 1. 6-hour aggregation
    sectionTitle(doc, "6-Hour Aggregation");
    drawTable(doc,
      ["Period", "Readings", "Avg Temp (°C)", "Avg Vib (mm/s)", "Avg RPM", "Avg Current (A)", "Faults"],
      aggregate6h(rows).map((b) => [
        b.label, b.count,
        fmt(b.avg_temp), fmt(b.avg_vib), fmt(b.avg_rpm, 0), fmt(b.avg_cur),
        b.faults > 0 ? `! ${b.faults}` : "0",
      ]),
      [110, 55, 75, 75, 58, 78, 64],
    );

    // 2. Fault readings
    sectionTitle(doc, "Fault / Deviated Readings");
    if (faults.length === 0) {
      doc.font("Helvetica").fontSize(9).fillColor("#555555").text("No fault readings recorded.", MARGIN).moveDown(0.3);
    } else {
      drawTable(doc,
        ["Time", "Temp (°C)", "Vib (mm/s)", "RPM", "Current (A)", "Status", "Risk Score"],
        faults.map((r) => [
          fmtTime(r.recorded_at),
          fmt(r.temperature_c), fmt(r.vibration_mm_s), fmt(r.rpm, 0), fmt(r.current_a),
          r.status ?? "—",
          `${riskScore(r)} / 100`,
        ]),
        [55, 60, 65, 50, 70, 65, 70],
      );
    }

    // 3. Line graphs
    sectionTitle(doc, "Sensor Graphs  (red dot/line = fault reading)");
    const charts = [
      { field: "temperature_c",  label: "Temperature", unit: "°C",   color: "#e67e22" },
      { field: "vibration_mm_s", label: "Vibration",   unit: "mm/s", color: "#2980b9" },
      { field: "rpm",            label: "RPM",         unit: "rpm",  color: "#27ae60" },
      { field: "current_a",      label: "Current",     unit: "A",    color: "#8e44ad" },
    ];
    
    for (const ch of charts) {
      // Adjusted height check to account for the image size properly
      if (doc.y + 190 > doc.page.height - 50) doc.addPage(); 
      try {
        const buf = await buildLineChart(rows, ch.field, ch.label, ch.unit, ch.color);
        doc.image(buf, MARGIN, doc.y, { width: COL_W });
        
        // FIX: Explicitly advance the text cursor so things don't overlap!
        doc.y += 175; 
      } catch {
        doc.font("Helvetica").fontSize(8).fillColor("red").text(`[Chart unavailable: ${ch.field}]`).moveDown(0.2);
      }
    }

    // 4. Anomaly table (high-risk faults only)
    sectionTitle(doc, "Anomaly Table");
    const anomalies = faults.filter((r) => riskScore(r) >= 55);
    if (anomalies.length === 0) {
      doc.font("Helvetica").fontSize(9).fillColor("#555555").text("No significant anomalies.", MARGIN).moveDown(0.3);
    } else {
      drawTable(doc,
        ["Time", "Temp (°C)", "Vib (mm/s)", "RPM", "Current (A)", "Status", "Risk"],
        anomalies.map((r) => [
          fmtTime(r.recorded_at),
          fmt(r.temperature_c), fmt(r.vibration_mm_s), fmt(r.rpm, 0), fmt(r.current_a),
          r.status ?? "—",
          `${riskScore(r)} / 100`,
        ]),
        [55, 65, 65, 50, 70, 65, 65],
      );
    }

    // 5. Summary
    sectionTitle(doc, "Summary");
    const summary = faults.length === 0
      ? `${machineId} operated normally throughout ${date}. All readings returned a healthy status — no faults detected.`
      : `${machineId} recorded ${faults.length} fault reading(s) on ${date}, between ` +
        `${fmtTime(faults[0].recorded_at)} and ${fmtTime(faults[faults.length - 1].recorded_at)}. ` +
        `Peak risk score: ${Math.max(...faults.map(riskScore))} / 100. ` +
        (anomalies.length > 0
          ? `${anomalies.length} high-severity anomaly(ies) require attention.`
          : "All faults were low-severity.");

    doc.font("Helvetica").fontSize(9).fillColor("#222222")
       .text(summary, MARGIN, doc.y, { width: COL_W }).moveDown(1.5); // Added padding at the bottom of the section

    drawHRule(doc);
    if (MACHINE_IDS.indexOf(machineId) < MACHINE_IDS.length - 1) doc.addPage();
  }

  // Footer
// Footer
// Footer
  const range = doc.bufferedPageRange();
  for (let i = range.start; i < range.start + range.count; i++) {
    doc.switchToPage(i);
    
    // THE FIX: Temporarily drop the safety margin to 0 so PDFKit doesn't panic
    const originalBottom = doc.page.margins.bottom;
    doc.page.margins.bottom = 0;

    doc.font("Helvetica").fontSize(7.5).fillColor("#aaaaaa")
       .text(
         `Generated ${new Date().toISOString()}  |  Page ${i + 1} of ${range.count}`,
         MARGIN, 
         doc.page.height - 30, 
         { 
           width: COL_W, 
           align: "center",
           lineBreak: false 
         }
       );

    // Restore the margin 
    doc.page.margins.bottom = originalBottom;
  }
  doc.end();
  await new Promise((res) => stream.on("finish", res));
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────
(async () => {
  const date     = getTargetDate();
  const dayStart = `${date}T00:00:00.000Z`;
  const dayEnd   = `${date}T23:59:59.999Z`;

  console.log(`\n📋  Generating report for ${date} …`);

  const fetchFn = (...args) => import("node-fetch").then(({ default: f }) => f(...args));

  const supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
    global: { fetch: fetchFn },
  });

  const { data, error } = await supabase
    .from("machine_telemetry")
    .select("*")
    .gte("recorded_at", dayStart)
    .lte("recorded_at", dayEnd)
    .order("recorded_at", { ascending: true });

  if (error) {
    console.error("❌  Supabase query failed:", error.message);
    process.exit(1);
  }

  console.log(`    Fetched ${data.length} rows across all machines.`);

  const outputPath = path.join(process.cwd(), `report_${date}.pdf`);
  await buildPDF(date, data, outputPath);
  console.log(`✅  Report saved → ${outputPath}\n`);
})();