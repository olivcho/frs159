import { useState } from 'react'
import './App.css'

function App() {

  return (
    <>
      <div className="flex flex-col items-center justify-center h-screen">
        <h1 className="text-2xl font-bold">Google Translate</h1>
        <div className="flex flex-row items-center justify-center">
          <input type="text" placeholder="Enter your text" className="w-64 px-4 py-2 border border-gray-300 rounded" />
          <button className="px-4 py-2 bg-blue-500 text-white rounded">Translate</button>
        </div>
      </div>
    </>
  )
}

export default App
