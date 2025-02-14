import { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import { validateURL } from "../services/validation";

export default async function complianceRoutes(fastify: FastifyInstance) {
  fastify.post(
    "/validate",
    async (request: FastifyRequest, reply: FastifyReply) => {
      const { url } = request.body as { url: string };

      if (!url || !validateURL(url)) {
        return reply.status(400).send({ error: "Invalid URL format" });
      }

      return reply.status(200).send({ message: "Valid input" });
    }
  );
}
