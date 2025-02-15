import axios from "axios";

/**
 * Calls the Python compliance checker API.
 * @param webpageText The text extracted from the webpage.
 * @param policyText The compliance policy text.
 * @returns A promise resolving to an array of non-compliant findings.
 */
export async function checkComplianceWithPython(
  webpageText: string,
  policyText: string,
  mode: "gemini" | "python"
): Promise<string[]> {
  try {
    console.log("length of webpageText", webpageText.length);
    console.log("length of policyText", policyText.length);
    const response = await axios.post(
      `http://python-server:5000/check_compliance${
        mode === "gemini" ? "_gemini" : ""
      }`,
      {
        webpageText,
        policyText,
      },
      {
        timeout: 0,
      }
    );

    return response.data.nonCompliantResults;
  } catch (error) {
    console.error("Error calling Python compliance API:", error);
    return ["Error analyzing compliance"];
  }
}
