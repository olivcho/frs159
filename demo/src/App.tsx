import { useState, useEffect } from 'react'
import './App.css'

function App() {

  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')

  const handleTranslate = async (input: string) => {
    if (!input.trim()) {
      setOutputText('');
      return;
    }

    const response = await fetch(
      `https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=yo&dt=t&q=${encodeURIComponent(input)}`
    );
    const data = await response.json();
    const translatedText = data[0]?.map((item: any[]) => item[0]).join('') || '';
    setOutputText(translatedText);
  }

  useEffect(() => {
    handleTranslate(inputText)
  }, [inputText])

  return (
    <div className="min-h-screen bg-white p-8">
      <div className="max-w-6xl mx-auto space-y-12">
        {/* Google Translate Section */}
        <div className="flex flex-col space-y-6">
          <h2 className="text-2xl font-bold text-geist-foreground">
            Google Translate (English to Yoruba)
          </h2>
          <div className="grid grid-cols-2 gap-6">
            <textarea
              placeholder="Enter your text"
              className="rounded-md resize-none font-sans bg-background-100 text-geist-foreground placeholder:text-gray-900 outline-none w-full duration-150 border border-gray-alpha-400 hover:border-gray-alpha-500 hover:ring-0 px-4 py-3 min-h-[120px]"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
            <div className="rounded-md font-sans bg-background-100 text-geist-foreground w-full duration-150 border border-gray-alpha-400 px-4 py-3 min-h-[120px] flex items-start">
              <span className={outputText ? '' : 'text-gray-400'}>
                {outputText || 'Translation will appear here...'}
              </span>
            </div>
          </div>
        </div>

        {/* LSTM Translate Section */}
        <div className="flex flex-col space-y-6">
          <h2 className="text-2xl font-bold text-geist-foreground">
            Bidirectional LSTM Translate w/ Diacritics (English to Yoruba)
          </h2>
          <div className="grid grid-cols-2 gap-6">
            <textarea
              placeholder="Enter your text"
              className="rounded-md resize-none font-sans bg-background-100 text-geist-foreground placeholder:text-gray-900 outline-none w-full duration-150 border border-gray-alpha-400 hover:border-gray-alpha-500 hover:ring-0 px-4 py-3 min-h-[120px]"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
            <div className="rounded-md font-sans bg-background-100 text-geist-foreground w-full duration-150 border border-gray-alpha-400 px-4 py-3 min-h-[120px] flex items-start">
              <span className={outputText ? '' : 'text-gray-400'}>
                {outputText || 'Translation will appear here...'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
