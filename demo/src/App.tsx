import { useState } from 'react'
import './App.css'

const API_URL = 'http://0.0.0.0:3001'

function App() {
  const [inputText, setInputText] = useState('')
  const [googleOutput, setGoogleOutput] = useState('')
  const [lstmOutput, setLstmOutput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleTranslate = async () => {
    if (!inputText.trim()) {
      setGoogleOutput('')
      setLstmOutput('')
      return
    }

    setIsLoading(true)
    setError('')
    let translatedText = ''

    // Step 1: Google Translate (English to Yoruba without diacritics)
    try {
      const googleResponse = await fetch(
        `https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=yo&dt=t&q=${encodeURIComponent(inputText)}`
      )
      const googleData = await googleResponse.json()
      translatedText = googleData[0]?.map((item: any[]) => item[0]).join('') || ''
      setGoogleOutput(translatedText)
    } catch (err) {
      console.error('Google Translate error:', err)
      setGoogleOutput('Translation error')
      setIsLoading(false)
      return
    }

    // Step 2: LSTM Diacritic Restoration (add diacritics to Google's output)
    try {
      const response = await fetch(`${API_URL}/restore`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: translatedText }),
      })
      const data = await response.json()
      // Remove <UNK> tokens from the output
      const cleanedOutput = (data.restored || 'Error restoring diacritics').replace(/<UNK>/g, '').replace(/\s+/g, ' ').trim()
      setLstmOutput(cleanedOutput)
    } catch (err) {
      console.error('LSTM API error:', err)
      setError('Backend not running. Start with: python app.py')
      setLstmOutput('')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-white p-8">
      <div className="max-w-6xl mx-auto space-y-12">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-geist-foreground">
            English to Yoruba Translation with Diacritic Restoration
          </h1>
          <p className="text-gray-600">
            Improving Google Translate with our Bi-LSTM diacritic restoration model
          </p>
        </div>

        {/* Input Section */}
        <div className="flex flex-col space-y-4">
          <label className="text-lg font-semibold text-geist-foreground">
            Enter English text:
          </label>
          <textarea
            placeholder="e.g., The child went to the house"
            className="rounded-md resize-none font-sans bg-background-100 text-geist-foreground placeholder:text-gray-400 outline-none w-full duration-150 border border-gray-alpha-400 hover:border-gray-alpha-500 hover:ring-0 px-4 py-3 min-h-[120px]"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
          <button
            onClick={handleTranslate}
            disabled={isLoading || !inputText.trim()}
            className="self-start px-6 py-3 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? 'Processing...' : 'Translate & Restore'}
          </button>
          {error && (
            <p className="text-red-500 text-sm">{error}</p>
          )}
        </div>

        {/* Results Grid */}
        <div className="grid grid-cols-2 gap-8">
          {/* Google Translate Section */}
          <div className="flex flex-col space-y-4">
            <h2 className="text-xl font-bold text-geist-foreground">
              Step 1: Google Translate
            </h2>
            <p className="text-sm text-gray-600">
              English → Yoruba (without proper diacritics)
            </p>
            <div className="rounded-md font-sans bg-background-100 text-geist-foreground w-full duration-150 border border-gray-alpha-400 px-4 py-3 min-h-[120px] flex items-start">
              <span className={googleOutput ? '' : 'text-gray-400'}>
                {googleOutput || 'Translation will appear here...'}
              </span>
            </div>
          </div>

          {/* LSTM Diacritic Restoration Section */}
          <div className="flex flex-col space-y-4">
            <h2 className="text-xl font-bold text-geist-foreground">
              Step 2: Bi-LSTM Diacritic Restoration
            </h2>
            <p className="text-sm text-gray-600">
              Adding proper tonal marks to improve accuracy
            </p>
            <div className="rounded-md font-sans bg-background-100 text-geist-foreground w-full duration-150 border-2 border-green-500 px-4 py-3 min-h-[120px] flex items-start">
              <span className={lstmOutput ? '' : 'text-gray-400'}>
                {lstmOutput || 'Restored text will appear here...'}
              </span>
            </div>
          </div>
        </div>

        {/* Info */}
        <div className="text-center text-sm text-gray-500 space-y-2">
          <p className="font-semibold">
            How it works:
          </p>
          <p>
            1. Google Translate converts English to Yoruba but lacks proper diacritics<br/>
            2. Our Bi-LSTM model adds tonal diacritics (á, à, ẹ, ọ, ṣ, etc.) to create accurate Yoruba text
          </p>
        </div>
      </div>
    </div>
  )
}

export default App
