from openai import OpenAI
from together import Together
import os
import anthropic
from IPython.display import display, Markdown, Latex
from google import genai
from google.genai import types
import copy


def make_clients():

    try:
        g_client = OpenAI()
    except:
        print("OPENAI_API_KEY not detected: Running OpenAI models will fail")
        g_client = None

    try:
        t_client = Together(api_key=os.environ['TOGETHER_API_KEY'])
    except:
        print("TOGETHER_API_KEY not detected: Running Together models will fail")
        t_client = None

    try:
        d_client = OpenAI(
            api_key=os.environ['DEEPSEEK_API_KEY'],
            base_url="https://api.deepseek.com",
        )
    except:
        print("DEEPSEEK_API_KEY not detected: Running deepseek models will fail")
        d_client = None

    try: 
        gen_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    except:
        print("GEMINI_API_KEY not detected: Running Gemini models will fail")
        gen_client = None

    try:
        c_client = anthropic.Anthropic()
    except:
        print("ANTHROPIC_API_KEY not detected: Running anthropic models will fail")
        c_client = None

    try:
        hf_client = OpenAI(
            base_url="https://router.huggingface.co/nebius/v1",
            api_key=os.environ['HF_API_KEY'],
        )
    except:
        print("HF_API_KEY not detected: Running huggingface endpoint models will fail")
        hf_client = None

    return g_client, t_client, d_client, gen_client, c_client, hf_client

g_client, t_client, d_client, gen_client, c_client, hf_client = make_clients()

def single_message(model, library, message=None, chain=None, sys=""):
    
    
    def single_message_gpt(send_chain):
        completion = g_client.chat.completions.create(
            model=model,
            messages=send_chain,
        )
        
        return completion.choices[0].message.content
        
        
    
    def single_message_together(send_chain):
        response = t_client.chat.completions.create(
            model=model,
            messages= send_chain,
        )
    
        return response.choices[0].message.content
    
    def single_message_hf(send_chain):
        response = hf_client.chat.completions.create(
            model=model,
            messages= send_chain,
        )
    
        return response.choices[0].message.content
    
    def single_message_deepseek(send_chain):
        completion = d_client.chat.completions.create(
            model=model, 
            messages=send_chain,
        )
        
        return completion.choices[0].message.content

    def single_message_anthropic(send_chain):

        system = [mes for mes in send_chain if mes['role'] == 'system']
        rest = [mes for mes in send_chain if mes['role'] != 'system']

        completion = c_client.messages.create(
            system=system[0]['content'],
            model=model,
            max_tokens=1024,
            messages=rest,
        )
        return completion.content[0].text
    
    def single_message_gemini(send_chain):

        system = [mes for mes in send_chain if mes['role'] == 'system']
        rest = [mes for mes in send_chain if mes['role'] != 'system']

        response = gen_client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
        system_instruction=system[0]['content']),
            contents=rest[-1]['content'],
        )

        return response.text
    
    send_chain = []
    
    if message is None and chain is None:
        raise Exception('single_message_together: No Input Provided')
    elif chain is None:
        send_chain = [
            {'role': 'system', 'content': sys},
            {"role": "user", "content": message},
        ]
    else:
        send_chain = chain
    
    if library == 'OA':
        return single_message_gpt(send_chain)
    elif library == 'deepseek':
        return single_message_deepseek(send_chain)
    elif library == 'anthropic':
        return single_message_anthropic(send_chain)
    elif library == 'GEM':
        return single_message_gemini(send_chain)
    elif library == 'together':
        return single_message_together(send_chain)     
    else:
        print("hep")
        return single_message_hf(send_chain)

class llmchain():
    
    def __init__(self, model=None, library=None, sys_message=""):
        self.library = library
        self.model = model
        self.sys_message = sys_message
        self.reschain = [
            {'role': 'system', 'content': self.sys_message}
        ]
        
    def change_model_lib(self, model, library):
        self.model = model
        self.library = library
        
    def clear_chain(self):
        self.reschain = [
            {'role': 'system', 'content': self.sys_message}
        ]
        
    def set_sys_message(self, sys_mes):
        self.sys_message = sys_mes
        self.reschain[0]['content'] = self.sys_message
        
        
    def append_chat(self, message):
        self.reschain.append(
            {
                'role': 'user',
                'content': message, 
            }
        )
        response = single_message(self.model, self.library, chain=self.reschain)
        self.reschain.append(
            {
                'role':'assistant',
                'content': response,
            }
        )
        return response

    def chat(self, message):

        reschain_copy = copy.deepcopy(self.reschain)
        reschain_copy.append(
            {
                'role': 'user',
                'content': message, 
            }
        )
        response = single_message(self.model, self.library, chain=reschain_copy)
        return response
    
        
    
def mark_print(text):
    display(Markdown(text))