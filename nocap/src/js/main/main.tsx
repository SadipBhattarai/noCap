import { useState } from "react";
import { evalTS } from "../lib/utils/bolt";
import { http } from "../lib/cep/node";
import "./main.scss";

type CaptionResponse = {
  status?: string;
  srt_path?: string;
  captions?: { start: number; end: number; text: string }[];
  detail?: string;
  transcription_provider?: string;
};

const getFileName = (filePath: string): string =>
  filePath.split(/[\\/]/).pop() || filePath;

const postCaptions = (filePath: string): Promise<CaptionResponse> => {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({ file_path: filePath });
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: 8000,
        path: "/generate-captions",
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
        },
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => {
          data += chunk;
        });
        res.on("end", () => {
          let parsed: CaptionResponse = {};
          try {
            parsed = data ? JSON.parse(data) : {};
          } catch (err) {
            return reject(
              new Error(data || "Invalid response from caption server"),
            );
          }
          if (
            !res.statusCode ||
            res.statusCode < 200 ||
            res.statusCode >= 300
          ) {
            return reject(
              new Error(
                parsed.detail || `Caption server error ${res.statusCode}`,
              ),
            );
          }
          resolve(parsed);
        });
      },
    );
    req.on("error", (err: any) =>
      reject(new Error(`Server request failed: ${err.message}`)),
    );
    req.write(body);
    req.end();
  });
};

export const App = () => {
  const [status, setStatus] = useState("Ready to generate captions");
  const [isLoading, setIsLoading] = useState(false);

  const generateCaptions = async () => {
    setIsLoading(true);
    setStatus("Getting selected clip path from Premiere...");

    try {
      const clipRes: any = await evalTS("getSelectedClipPath");

      if (
        !clipRes ||
        clipRes.error ||
        typeof clipRes.path !== "string" ||
        !clipRes.path
      ) {
        throw new Error(
          clipRes?.error ||
            "Premiere did not return a valid path for the selected clip.",
        );
      }

      const clipPath = clipRes.path;
      const clipStart =
        typeof clipRes.startSeconds === "number" &&
        Number.isFinite(clipRes.startSeconds)
          ? clipRes.startSeconds
          : 0;

      setStatus(`Transcribing audio for: ${getFileName(clipPath)}...`);

      const result = await postCaptions(clipPath);

      if (result.status !== "success" || !result.srt_path) {
        throw new Error("Caption server did not return a valid SRT path.");
      }

      setStatus("Importing captions into Premiere Pro...");

      const importResult = await evalTS(
        "importSRT",
        result.srt_path,
        clipStart,
      );

      if (importResult !== "SUCCESS") {
        throw new Error(
          String(
            importResult || "Premiere rejected the caption track placement.",
          ),
        );
      }

      const providerNotice =
        result.transcription_provider === "local-whisper-fallback"
          ? " Gemini was unavailable; local Whisper was used instead."
          : "";
      setStatus(
        `Success! Captions placed on the timeline.${providerNotice}`,
      );
    } catch (err: any) {
      console.error(err);
      setStatus(
        `Error: ${
          err && typeof err === "object" && "message" in err
            ? String(err.message)
            : String(err)
        }`,
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className="app"
      style={{
        backgroundColor: "#1e1e1e",
        color: "#ffffff",
        padding: "20px",
        fontFamily: "sans-serif",
      }}
    >
      <h2>noCap Auto-Captions</h2>
      <p style={{ fontSize: "13px", color: "#b0b0b0" }}>
        Select a clip on your timeline and click below to generate & import AI
        captions.
      </p>
      <div style={{ marginTop: "20px" }}>
        <button
          onClick={generateCaptions}
          disabled={isLoading}
          style={{
            background: isLoading ? "#555" : "#0066cc",
            color: "#fff",
            border: "none",
            padding: "10px 20px",
            borderRadius: "4px",
            cursor: isLoading ? "not-allowed" : "pointer",
            fontWeight: "bold",
            width: "100%",
          }}
        >
          {isLoading ? "Processing..." : "Generate Auto-Captions"}
        </button>
      </div>
      <div
        style={{
          marginTop: "20px",
          background: "#2d2d2d",
          padding: "12px",
          borderRadius: "4px",
        }}
      >
        <strong style={{ fontSize: "12px", color: "#888" }}>STATUS:</strong>
        <p
          style={{
            fontSize: "13px",
            margin: "6px 0 0 0",
            wordBreak: "break-word",
          }}
        >
          {status}
        </p>
      </div>
    </div>
  );
};
