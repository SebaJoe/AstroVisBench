# Generate Code

Here are scripts you can you use to query major LLM APIs to fill the benchmark.

Currently, this script supports API access to:
- OpenAI
- Together.AI
- Gemini
- Anthropic
- Huggingface Endpoints

Make sure you have your API keys loaded in your enviornment variables before running this script. 

To use the script, simply execute the following command:
```
python generate_code.py \
<INPUT BENCHMARK JSON>
<OUTFILE>
<MODEL NAME> # Name should match with the name used by the API
<LIBRARY> # OA for OpenAI, anthropic for Anthropic, together for Together.AI, GEM for Gemini, hf for Huggingface Endpoints
--generate-pro-only # Only complete the processing sections
--generate-vis-only # Only complete the visualization sections
--generate-vis-loose # Use in tandem with --generate-pro-only; fill only when visualization_gen_code is empty.
```


