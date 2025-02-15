import { FastifyReply, FastifyRequest, HookHandlerDoneFunction } from "fastify";

/**
 * Checks if a given URL is valid.
 * @param url - The URL string to validate
 * @returns True if the URL is valid, otherwise false
 */
export function validateURL(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Middleware to validate webpageUrl and policyUrl in the request body.
 * Returns a detailed error response for each invalid case.
 */
export function validateURLMiddleware(
  req: FastifyRequest,
  reply: FastifyReply,
  done: HookHandlerDoneFunction
) {
  const { webpageUrl, policyUrl } = req.body as {
    webpageUrl?: string;
    policyUrl?: string;
  };

  // Check if both fields are present
  if (!webpageUrl && !policyUrl) {
    return reply.status(400).send({
      error:
        "Both 'webpageUrl' and 'policyUrl' are required in the request body.",
    });
  }

  if (!webpageUrl) {
    return reply.status(400).send({
      error: "'webpageUrl' is missing in the request body.",
    });
  }

  if (!policyUrl) {
    return reply.status(400).send({
      error: "'policyUrl' is missing in the request body.",
    });
  }

  // Validate URL formats
  if (!validateURL(webpageUrl)) {
    return reply.status(400).send({
      error: `'webpageUrl' is invalid. Ensure it is a properly formatted URL (e.g., 'https://example.com').`,
    });
  }

  if (!validateURL(policyUrl)) {
    return reply.status(400).send({
      error: `'policyUrl' is invalid. Ensure it is a properly formatted URL (e.g., 'https://example.com').`,
    });
  }

  return done();
}
