const { config } = require("dotenv");
const { OpenAI } = require("openai");
const { readFile } = require("fs/promises");

config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

function dotProduct(vectorA, vectorB) {
  return vectorA.reduce((sum, a, idx) => sum + a * vectorB[idx], 0);
}

function magnitude(vector) {
  return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
}

function cosineSimilarity(vectorA, vectorB) {
  const dotProd = dotProduct(vectorA, vectorB);
  const magnitudeA = magnitude(vectorA);
  const magnitudeB = magnitude(vectorB);
  return dotProd / (magnitudeA * magnitudeB);
}

(async () => {
  const art = await readFile("./art.txt");
  const weatherart = await readFile("./weatherart.txt");
  const weather = await readFile("./weather.txt");

  const artEmbedding = await openai.embeddings.create({
    model: "text-embedding-3-large",
    input: art.toString(),
  });

  const weatherartEmbedding = await openai.embeddings.create({
    model: "text-embedding-3-large",
    input: weatherart.toString(),
  });

  const weatherEmbedding = await openai.embeddings.create({
    model: "text-embedding-3-large",
    input: weather.toString(),
  });

  const question = "What is the weather?";

  const questionEmbedding = await openai.embeddings.create({
    model: "text-embedding-3-large",
    input: question,
  });

  console.log(
    cosineSimilarity(
      questionEmbedding.data[0].embedding,
      artEmbedding.data[0].embedding
    )
  );
  console.log(
    cosineSimilarity(
      questionEmbedding.data[0].embedding,
      weatherartEmbedding.data[0].embedding
    )
  );
  console.log(
    cosineSimilarity(
      questionEmbedding.data[0].embedding,
      weatherEmbedding.data[0].embedding
    )
  );
})();
