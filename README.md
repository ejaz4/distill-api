# distill-api

General-purpose Express API that converts mixed content into structured cards using OpenRouter.

## Setup

1) Install dependencies
2) Copy [.env.example](.env.example) to `.env` and set `EXPO_PUBLIC_OPENROUTER_API_KEY`

## Run

`npm run dev`

Server starts on `PORT` (default 3000).

## Endpoints

### POST /cards

**Body**

```json
{
	"pageContent": [
		{ "type": "text", "content": "Some text" },
		{ "type": "image", "src": "https://example.com/photo.jpg" }
	],
	"model": "meta-llama/llama-3.1-8b-instruct"
}
```

**Response**

```json
{
	"cards": [
		{
			"name": "Example",
			"image": "https://example.com/photo.jpg",
			"fields": [{ "label": "Note", "value": "Sample" }],
			"buttons": [{ "label": "Visit", "action": "navigate", "url": "https://example.com" }]
		}
	]
}
```

### GET /health

Returns `{ "ok": true }`.