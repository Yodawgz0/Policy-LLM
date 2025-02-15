import axios from "axios";
import * as cheerio from "cheerio";

/**
 * Fetches and cleans text content from a given webpage URL.
 * @param url The webpage URL
 * @returns A single cleaned condensed string
 */
export async function fetchPageText(url: string): Promise<string> {
  try {
    const { data } = await axios.get(url, {
      headers: { "User-Agent": "Mozilla/5.0" }, // Avoid bot detection
    });

    const $ = cheerio.load(data);
    let textContent = "";

    // Extract text from relevant elements and clean it
    $("p, h1, h2, h3, h4, h5, h6, li, span, div").each((_, el) => {
      let text = $(el).text().trim();

      // Remove multiple spaces and newlines
      text = text.replace(/\s+/g, " ");

      // Remove links (URLs)
      text = text.replace(/https?:\/\/[^\s]+/g, "");

      // Remove brackets and their content
      text = text.replace(/[\[\]{}()]/g, "");

      // Remove single-digit numbers
      text = text.replace(/\b\d\b/g, "");

      // Remove special symbols (©, ®, $, %, &, *, etc.)
      text = text.replace(/[©®™$%^&*<>+=#~@]/g, "");

      // Remove backslashes
      text = text.replace(/\\/g, "");

      // Remove quotes
      text = text.replace(/["']/g, "");

      if (text.length > 0) {
        textContent += text + ". "; // Ensure proper punctuation
      }
    });

    // Remove duplicate words in condensed text
    textContent = textContent
      .split(". ")
      .filter((line, index, self) => self.indexOf(line) === index) // Remove duplicate sentences
      .join(". ");

    return textContent.trim(); // Return final cleaned and condensed text
  } catch (error) {
    console.error("Error fetching page:", error);
    throw new Error("Failed to fetch page content.");
  }
}
