import express from "express";
import cors from "cors";
import { z } from "zod";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { generateObject } from "ai";
import { JSDOM } from "jsdom";

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// ============================================================================
// Distillation System Prompt
// ============================================================================
const DISTILLATION_SYSTEM_PROMPT = `You are an expert content distillation AI for Distill, an app that transforms complex web pages into beautiful, scannable UI components.

AVAILABLE UI COMPONENT TYPES:
1. cards - Reviews, comparisons, product listings, portfolios, team pages, feature lists
2. article - Long-form content, stories, blog posts, news articles, tutorials
3. timeline - Chronological events, step-by-step processes, historical content, how-to guides
4. comparison - Side-by-side comparisons, specs, pricing tables, feature matrices
5. faq - FAQ pages, Q&A format, troubleshooting, interview-style content
6. stats - Data-heavy pages, research findings, statistics, reports, infographics
7. hero - Landing pages, product pages, event pages, announcements, single-focus content
8. list - Top 10 lists, checklists, tips, recommendations, rankings
9. gallery - Image-heavy content, portfolios, photo essays, visual stories
10. profile - Person profiles, company about pages, artist bios, author pages
11. quote - Inspirational content, key insights, single powerful messages, testimonials

CRITICAL - IMAGE EXTRACTION:
The input contains both text AND image items. You MUST extract and use images from the input:
- Look for items with type: "image" and src: "..." in the input array
- Match images to relevant cards/content by analyzing surrounding text context
- For cards: Use "banner" field for a small header image strip on each card
- For cards: Use "image" field for full-size background images (use sparingly for hero cards)
- For hero components: Always include heroImage if a relevant image exists
- For gallery: Extract all relevant images
- For profiles: Include avatar/photo images
- NEVER leave cards without images if relevant images exist in the input
- Images make the output visually engaging - prioritize using them!

MULTI-COMPONENT STRATEGY:
- Combine MULTIPLE component types to create a rich, visually diverse distilled output
- Components render top to bottom in the order you specify
- Use different components to highlight different aspects of the content
- Create visual variety and maintain user engagement
- Start with the most important/engaging component (often hero or quote)
- Typically use 1-4 components depending on content richness

SELECTION CRITERIA:
- Choose components that BEST represent the content's information architecture
- Prioritize user comprehension and visual appeal
- Consider the content's natural structure and hierarchy
- Use images strategically throughout components
- Mix text-heavy and visual components for balance
- Extract the ESSENCE - distill, don't just summarize

QUALITY GUIDELINES:
- Be concise but comprehensive
- Preserve accuracy of facts, numbers, and quotes
- Use markdown in content fields (article, descriptions) where appropriate
- ALWAYS include image URLs from the input when available - this is critical for visual appeal
- Create visual variety - avoid using the same component type multiple times
- Order components from most engaging to supporting details`;

// ============================================================================
// URL Content Extraction
// ============================================================================
function isLikelyNoiseElement(el) {
  const tagName = el.tagName?.toLowerCase() || '';
  if (['script', 'style', 'noscript', 'iframe', 'svg', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'input', 'select', 'textarea'].includes(tagName)) {
    return true;
  }
  const id = (el.id || '').toLowerCase();
  const className = (el.className || '').toString().toLowerCase();
  const noiseTokens = ['ad', 'ads', 'advert', 'promo', 'cookie', 'banner', 'subscribe', 'newsletter', 'share', 'social', 'sidebar', 'footer', 'header', 'nav', 'menu', 'popup', 'modal'];
  return noiseTokens.some(token => id.includes(token) || className.includes(token));
}

async function extractContentFromUrl(url) {
  // Fetch the page
  const response = await fetch(url, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.5',
    },
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch URL: ${response.status} ${response.statusText}`);
  }
  
  const html = await response.text();
  const dom = new JSDOM(html);
  const document = dom.window.document;
  
  const content = [];
  
  // Find main content container
  const candidates = Array.from(document.querySelectorAll('main, article, [role="main"], #content, .content, .article, .post, .entry-content'));
  const filtered = candidates.filter(el => !isLikelyNoiseElement(el));
  
  let main;
  if (filtered.length > 0) {
    main = filtered.reduce((best, el) => 
      (el.textContent?.length || 0) > (best.textContent?.length || 0) ? el : best
    );
  } else {
    main = document.body;
  }
  
  function extractFromElement(el) {
    if (!el || isLikelyNoiseElement(el)) return;
    
    const tagName = el.tagName?.toLowerCase();
    
    if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
      const text = el.textContent?.trim();
      if (text) content.push({ type: 'text', content: text });
      return;
    }
    
    if (tagName === 'img') {
      let src = el.getAttribute('src') || el.getAttribute('data-src');
      if (src) {
        // Make relative URLs absolute
        try {
          src = new URL(src, url).href;
        } catch {}
        content.push({ type: 'image', src });
      }
      return;
    }
    
    if (tagName === 'p') {
      const text = el.textContent?.trim();
      if (text && text.length > 20) {
        content.push({ type: 'text', content: text });
      }
      return;
    }
    
    // Recurse into children
    if (el.children) {
      Array.from(el.children).forEach(extractFromElement);
    }
  }
  
  extractFromElement(main);
  
  return content;
}

// ============================================================================
// Validation Schemas
// ============================================================================
const contentItemSchema = z.union([
  z.object({
    type: z.literal("text"),
    content: z.string(),
  }),
  z.object({
    type: z.literal("image"),
    src: z.string(),
  }),
]);

const requestSchema = z.object({
  items: z.array(contentItemSchema).min(1),
  model: z.string().optional(),
});

// ============================================================================
// Component Schemas for AI Response
// ============================================================================
const specItemSchema = z.object({
  key: z.string(),
  value: z.string(),
});

const cardDataSchema = z.object({
  title: z.string(),
  subtitle: z.string().optional(),
  image: z.string().optional().describe("Full card image URL for prominent visual cards"),
  banner: z.string().optional().describe("Small banner image URL from page for cards with a header image strip"),
  icon: z.string().optional().describe("Icon name (e.g. 'chart', 'users', 'money', 'star') for icon-style cards"),
  rating: z.number().optional(),
  price: z.string().optional(),
  pros: z.array(z.string()).optional(),
  cons: z.array(z.string()).optional(),
  summary: z.string().optional(),
  specs: z.array(specItemSchema).optional(),
  tags: z.array(z.string()).optional(),
});

const cardsComponentSchema = z.object({
  type: z.literal("cards"),
  title: z.string(),
  description: z.string().optional(),
  cards: z.array(cardDataSchema),
});

const articleComponentSchema = z.object({
  type: z.literal("article"),
  title: z.string(),
  subtitle: z.string().optional(),
  author: z.string().optional(),
  readTime: z.string().optional(),
  heroImage: z.string().optional(),
  content: z.string(),
  keyTakeaways: z.array(z.string()).optional(),
});

const timelineItemSchema = z.object({
  date: z.string().optional(),
  title: z.string(),
  description: z.string().optional(),
  image: z.string().optional(),
  icon: z.string().optional(),
});

const timelineComponentSchema = z.object({
  type: z.literal("timeline"),
  title: z.string(),
  description: z.string().optional(),
  orientation: z.enum(["vertical", "horizontal"]).optional(),
  items: z.array(timelineItemSchema),
});

const comparisonCategorySchema = z.object({
  name: z.string(),
  values: z.array(z.string()),
  highlight: z.number().optional(),
});

const comparisonComponentSchema = z.object({
  type: z.literal("comparison"),
  title: z.string(),
  description: z.string().optional(),
  items: z.array(z.string()),
  categories: z.array(comparisonCategorySchema),
});

const faqItemSchema = z.object({
  question: z.string(),
  answer: z.string(),
  category: z.string().optional(),
});

const faqComponentSchema = z.object({
  type: z.literal("faq"),
  title: z.string(),
  description: z.string().optional(),
  questions: z.array(faqItemSchema),
});

const statItemSchema = z.object({
  value: z.string(),
  label: z.string(),
  trend: z.enum(["up", "down", "neutral", "warning"]).optional(),
  icon: z.string().optional(),
  context: z.string().optional(),
});

const chartDataSchema = z.object({
  type: z.enum(["bar", "line", "pie"]),
  title: z.string(),
  description: z.string().optional(),
});

const statsComponentSchema = z.object({
  type: z.literal("stats"),
  title: z.string(),
  summary: z.string().optional(),
  stats: z.array(statItemSchema),
  charts: z.array(chartDataSchema).optional(),
});

const heroHighlightSchema = z.object({
  icon: z.string().optional(),
  title: z.string(),
  description: z.string(),
});

const heroCtaSchema = z.object({
  text: z.string(),
  url: z.string(),
});

const heroComponentSchema = z.object({
  type: z.literal("hero"),
  headline: z.string(),
  subheadline: z.string().optional(),
  heroImage: z.string().optional(),
  cta: heroCtaSchema.optional(),
  highlights: z.array(heroHighlightSchema).optional(),
  summary: z.string().optional(),
});

const listItemSchema = z.object({
  rank: z.number().optional(),
  title: z.string(),
  description: z.string().optional(),
  image: z.string().optional(),
  tags: z.array(z.string()).optional(),
});

const listComponentSchema = z.object({
  type: z.literal("list"),
  title: z.string(),
  description: z.string().optional(),
  listStyle: z.enum(["numbered", "bulleted", "featured"]).optional(),
  items: z.array(listItemSchema),
});

const galleryImageSchema = z.object({
  url: z.string(),
  caption: z.string().optional(),
  credit: z.string().optional(),
  tags: z.array(z.string()).optional(),
});

const galleryComponentSchema = z.object({
  type: z.literal("gallery"),
  title: z.string(),
  description: z.string().optional(),
  layout: z.enum(["grid", "masonry", "carousel"]).optional(),
  images: z.array(galleryImageSchema),
});

const profileStatSchema = z.object({
  label: z.string(),
  value: z.string(),
});

const profileSocialSchema = z.object({
  twitter: z.string().optional(),
  linkedin: z.string().optional(),
  github: z.string().optional(),
  website: z.string().optional(),
}).passthrough();

const profileComponentSchema = z.object({
  type: z.literal("profile"),
  name: z.string(),
  title: z.string().optional(),
  image: z.string().optional(),
  tagline: z.string().optional(),
  bio: z.string().optional(),
  stats: z.array(profileStatSchema).optional(),
  highlights: z.array(z.string()).optional(),
  social: profileSocialSchema.optional(),
});

const quoteComponentSchema = z.object({
  type: z.literal("quote"),
  quote: z.string(),
  author: z.string().optional(),
  context: z.string().optional(),
  image: z.string().optional(),
  backgroundImage: z.string().optional(),
  relatedContent: z.string().optional(),
});

// Union of all component types
const distilledComponentSchema = z.discriminatedUnion("type", [
  cardsComponentSchema,
  articleComponentSchema,
  timelineComponentSchema,
  comparisonComponentSchema,
  faqComponentSchema,
  statsComponentSchema,
  heroComponentSchema,
  listComponentSchema,
  galleryComponentSchema,
  profileComponentSchema,
  quoteComponentSchema,
]);

// Main response schema
const distilledResponseSchema = z.object({
  components: z.array(distilledComponentSchema),
});

const buildContentForModel = (items) => {
  const content = items
    .map((item) => {
      if (item.type === "text") return item.content;
      return `[image: ${item.src}]`;
    })
    .join("\n");

  // eslint-disable-next-line no-console
  console.log("/api/cards: content built", {
    items: items.length,
    length: content.length,
  });

  return content;
};

const DEFAULT_MODEL = process.env.DEFAULT_MODEL ?? "google/gemini-2.0-flash-001";

const getOpenRouterModel = (model) => {
  const apiKey = process.env.OPENROUTER_API_KEY ?? "";
  if (!apiKey) {
    const error = new Error("Missing OPENROUTER_API_KEY");
    error.statusCode = 400;
    throw error;
  }
  const openrouter = createOpenRouter({ apiKey });
  return openrouter(model ?? DEFAULT_MODEL);
};

// Health check
app.get("/health", (_req, res) => {
  res.json({ ok: true });
});


// POST route for URL-based distillation (for web app)
app.post("/api/distill-url", async (req, res) => {
  // eslint-disable-next-line no-console
  console.log("/api/distill-url: request received");

  const { url, model } = req.body;
  
  if (!url || typeof url !== 'string') {
    return res.status(400).json({ error: "URL is required" });
  }

  try {
    // Extract content from URL
    // eslint-disable-next-line no-console
    console.log("/api/distill-url: extracting content from", url);
    const items = await extractContentFromUrl(url);
    
    if (items.length === 0) {
      return res.status(400).json({ error: "No content could be extracted from the page" });
    }
    
    // eslint-disable-next-line no-console
    console.log("/api/distill-url: extracted", items.length, "items");
    
    const contentForModel = buildContentForModel(items);
    
    // eslint-disable-next-line no-console
    console.log("/api/distill-url: generating distilled content");
    const result = await generateObject({
      model: getOpenRouterModel(model),
      mode: "json",
      schema: distilledResponseSchema,
      system: DISTILLATION_SYSTEM_PROMPT,
      prompt: `Analyze the following webpage content and distill it into the most appropriate UI components.
Select 1-4 component types that best represent this content's structure and information.
Order components from most engaging to supporting details.
Include image URLs from the content where relevant.

Content to distill:
${contentForModel}`,
    });

    // eslint-disable-next-line no-console
    console.log("/api/distill-url: generation complete", {
      components: result.object.components.length,
      types: result.object.components.map((c) => c.type),
    });

    return res.json(result.object);
  } catch (err) {
    const statusCode = err?.statusCode ?? 500;
    const message = err instanceof Error ? err.message : "Unknown error";
    // eslint-disable-next-line no-console
    console.log("/api/distill-url: error", { statusCode, message });
    return res.status(statusCode).json({ error: message });
  }
});


// POST route for distillation
app.post("/api/distill", async (req, res) => {
  // eslint-disable-next-line no-console
  console.log("/api/distill: request received");

  const parseResult = requestSchema.safeParse(req.body);
  if (!parseResult.success) {
    // eslint-disable-next-line no-console
    console.log("/api/distill: validation failed", parseResult.error.flatten());
    return res.status(400).json({
      error: "Invalid request body",
      details: parseResult.error.flatten(),
    });
  }

  const { items, model } = parseResult.data;
  // eslint-disable-next-line no-console
  console.log("/api/distill: validation passed", {
    items: items.length,
    model: model ?? DEFAULT_MODEL,
  });

  const contentForModel = buildContentForModel(items);

  try {
    // eslint-disable-next-line no-console
    console.log("/api/distill: generating distilled content");
    const result = await generateObject({
      model: getOpenRouterModel(model),
      mode: "json",
      schema: distilledResponseSchema,
      system: DISTILLATION_SYSTEM_PROMPT,
      prompt: `Analyze the following webpage content and distill it into the most appropriate UI components.
Select 1-4 component types that best represent this content's structure and information.
Order components from most engaging to supporting details.
Include image URLs from the content where relevant.

Content to distill:
${contentForModel}`,
    });

    // eslint-disable-next-line no-console
    console.log("/api/distill: generation complete", {
      components: result.object.components.length,
      types: result.object.components.map((c) => c.type),
    });

    return res.json(result.object);
  } catch (err) {
    const statusCode = err?.statusCode ?? 500;
    const message = err instanceof Error ? err.message : "Unknown error";
    // eslint-disable-next-line no-console
    console.log("/api/distill: error", {
      statusCode,
      message,
    });
    return res.status(statusCode).json({ error: message });
  }
});


// Legacy endpoint - redirect to new endpoint
app.post("/api/cards", async (req, res) => {
  // eslint-disable-next-line no-console
  console.log("/api/cards: legacy endpoint called, forwarding to /api/distill");
  
  const parseResult = requestSchema.safeParse(req.body);
  if (!parseResult.success) {
    return res.status(400).json({
      error: "Invalid request body",
      details: parseResult.error.flatten(),
    });
  }

  const { items, model } = parseResult.data;
  const contentForModel = buildContentForModel(items);

  try {
    const result = await generateObject({
      model: getOpenRouterModel(model),
      mode: "json",
      schema: distilledResponseSchema,
      system: DISTILLATION_SYSTEM_PROMPT,
      prompt: `Analyze the following webpage content and distill it into the most appropriate UI components.
Select 1-4 component types that best represent this content's structure and information.
Order components from most engaging to supporting details.
Include image URLs from the content where relevant.

Content to distill:
${contentForModel}`,
    });

    return res.json(result.object);
  } catch (err) {
    const statusCode = err?.statusCode ?? 500;
    const message = err instanceof Error ? err.message : "Unknown error";
    return res.status(statusCode).json({ error: message });
  }
});


// Error handler
app.use((err, _req, res, _next) => {
  // eslint-disable-next-line no-console
  console.error(err);
  res.status(500).json({ error: "Internal Server Error" });
});

const port = Number(process.env.PORT ?? 3000);
app.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`Server listening on port ${port}`);
});
