import { FastifyRequest, FastifyReply } from "fastify";
import { fetchPageText } from "../services/scraperService";
import { checkComplianceWithPython } from "../services/complianceService";

export const complianceApiController = async (
  request: FastifyRequest,
  reply: FastifyReply
) => {
  const { webpageUrl, policyUrl } = request.body as {
    webpageUrl: string;
    policyUrl: string;
  };

  try {
    const webpageText = await fetchPageText(webpageUrl);
    const policyText = await fetchPageText(policyUrl);

    const nonCompliantResults = await checkComplianceWithPython(
      webpageText,
      policyText
    );

    return reply
      .status(200)
      .send({ webpageUrl, policyUrl, nonCompliantResults });
  } catch (error) {
    return reply
      .status(500)
      .send({ error: "Error processing compliance check" });
  }
};
