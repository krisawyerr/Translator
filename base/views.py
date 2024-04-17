from transformers import MarianMTModel, MarianTokenizer
from django.shortcuts import render, redirect

def translate_to_spanish(text, lang):
    model_name = f"Helsinki-NLP/opus-mt-en-{lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    chunk_size = 500  
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    translated_chunks = []
    for chunk in chunks:
        inputs = tokenizer.encode(chunk, return_tensors="pt")
        translated_chunk = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
        translated_chunk = tokenizer.decode(translated_chunk[0], skip_special_tokens=True)
        translated_chunks.append(translated_chunk)

    translated_text = " ".join(translated_chunks)

    print("English Text:", text)
    print("Translated Text (Spanish):", translated_text)
    return translated_text

conversation = []

def translate(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '').strip().lower()
        conversation.append(('English', user_input))
        lang = 'es'
        if lang == 'en':
            spanish_text = user_input
            conversation.append(('Translation', spanish_text))
        else:
            spanish_text = translate_to_spanish(user_input, lang)
            conversation.append(('Translation', spanish_text))
        return render(request, 'base/home.html', {'conversation': conversation})
    else:
        return render(request, 'base/home.html')
    
def clearChat(request):
    conversation.clear()
    return redirect('translate')