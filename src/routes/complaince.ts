import { FastifyInstance } from "fastify";
import { validateURLMiddleware } from "../services/validation";
import { complianceApiController } from "../controllers/complainceApiController";

export default async function complianceRoutes(fastify: FastifyInstance) {
  fastify.post(
    "/api_compliance_check",
    { preHandler: validateURLMiddleware },
    complianceApiController
  );
}
