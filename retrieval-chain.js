// Import necessary modules from LangChain and dependencies
import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";

// Import environment variables handling library
import * as dotenv from "dotenv";
dotenv.config(); // Load environment variables from a file (e.g., API key)

// Instantiate OpenAI Chat Model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.9,
});

// Create a prompt template for the chat model
const prompt = ChatPromptTemplate.fromTemplate(
  `Answer the user's question from the following context: 
  {context}
  Question: {input}`
);

// Create a document chain using LangChain and the instantiated model
// allow us to pass in documents 
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});

// Manually create documents (commented out in this example)
// const documentA = new Document({ ... });
// const documentB = new Document({ ... });

// Use Cheerio to scrape content from a webpage and create documents
const loader = new CheerioWebBaseLoader("https://hiveclimbing.com/");
const docs = await loader.load();

// Use a text splitter to divide the documents into smaller chunks
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});
const splitDocs = await splitter.splitDocuments(docs);

// Instantiate an OpenAIEmbeddings function
const embeddings = new OpenAIEmbeddings();

// Create a vector store from the split documents
// gets the information from the documents, embed them and store it
const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

// Create a retriever from the vector store, specifying the number of retrieved documents (k: 2)
const retriever = vectorstore.asRetriever({ k: 2 });

// Create a retrieval chain by combining document retrieval with the previously created document chain
const retrievalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever,
});

// Invoke the retrieval chain with a specific input question and print the response
const response = await retrievalChain.invoke({
  input: "what is the color of the sea?",
});

const response2 = await retrievalChain.invoke({
  input: "what other colors are in there?",
});
console.log(response2);
