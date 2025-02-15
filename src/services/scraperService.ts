import axios from "axios";
import * as cheerio from "cheerio";

export async function fetchPageText(url: string): Promise<string> {
  try {
    const { data, headers } = await axios.get(url, {
      headers: { "User-Agent": "Mozilla/5.0" }, // Avoid bot detection
    });

    let textContent = "";

    // 🔹 Check if the response is JSON (e.g., Markdoc-based docs)
    if (headers["content-type"]?.includes("application/json")) {
      console.log("⚠️ Received JSON instead of HTML, extracting text...");

      // Extract text from the JSON response
      if (data.article?.content?.children) {
        textContent = extractTextFromJson(data.article.content.children);
      } else {
        console.error("❌ Unexpected JSON format");
        throw new Error("Invalid response structure");
      }
    } else {
      // 🔹 Process as HTML using Cheerio
      console.log("✅ Processing as HTML");
      const $ = cheerio.load(data);

      $("p, h1, h2, h3, h4, h5, h6, li, span, div").each((_, el) => {
        let text = $(el).text().trim();

        // Clean up text
        text = text.replace(/\s+/g, " ").replace(/https?:\/\/[^\s]+/g, "");
        text = text
          .replace(/[\[\]{}()©®™$%^&*<>+=#~@]/g, "")
          .replace(/\\/g, "");
        text = text.replace(/\b\d\b/g, "").replace(/["']/g, "");

        if (text.length > 0) {
          textContent += text + ". ";
        }
      });

      // Remove duplicate sentences
      textContent = textContent
        .split(". ")
        .filter((line, index, self) => self.indexOf(line) === index)
        .join(". ");
    }

    return textContent.trim();
  } catch (error) {
    console.error("Error fetching page:", error);
    throw new Error("Failed to fetch page content.");
  }
}

/**
 * Recursively extracts text from Markdoc JSON structure.
 */
function extractTextFromJson(nodes: any[]): string {
  let text = "";

  for (const node of nodes) {
    if (typeof node === "string") {
      text += node + " ";
    } else if (node.children) {
      text += extractTextFromJson(node.children);
    }
  }

  return text.trim();
}
