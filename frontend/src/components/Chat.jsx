import React, { useEffect, useRef } from 'react';
import TypingMessage from "./TypingMessage";

function Chat({ messages, isLoading }) {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Hàm ĐƠN GIẢN: chỉ render string
  const renderMessage = (msg, index) => {
    let textToRender = "";

    try {
        // Ưu tiên lấy text
        if (msg && typeof msg.text === "string") {
          textToRender = msg.text;
        } else if (msg && typeof msg.text === "object") {
          // Nếu là object → stringify có kiểm soát
          textToRender =
              msg.text?.answer ||
              msg.text?.message ||
              msg.text?.text ||
              JSON.stringify(msg.text);
        } else if (typeof msg === "string") {
          textToRender = msg;
        } else {
          textToRender = String(msg?.text || msg?.message || "");
        }
    } catch (err) {
        console.error(`⚠️ Error parsing message ${index}:`, err);
        textToRender = "[Error displaying message]";
    }

    // ✅ Đảm bảo textToRender là chuỗi
    if (typeof textToRender !== "string") {
        textToRender = String(textToRender);
    }

    return (
        <div key={index} className={`message ${msg.sender || "bot"}`}>
          <div className="message-content">
            {/* CHỈ HIỂN THỊ TEXT THUẦN - KHÔNG CÓ MARKDOWN */}
            {msg.sender === "bot" ? (
              <TypingMessage text={textToRender} />
            ) : (
              <div style={{
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                lineHeight: '1.5'
              }}>
                {textToRender}
              </div>
            )}

            {Array.isArray(msg.sources) && msg.sources.length > 0 && (
              <div className="sources">
                <p><strong>Sources:</strong></p>
                <ul>
                  {msg.sources.map((source, i) => {
                    // FIX: Xử lý score để tránh NaN
                    let scoreText = "";
                    
                    if (source.score) {
                      try {
                        const scoreStr = String(source.score);
                        
                        // Kiểm tra nếu đã là string có format (ví dụ: "82% ✅")
                        if (scoreStr.includes('%')) {
                          // Lấy số phần trăm trước ký tự %
                          const percentMatch = scoreStr.match(/(\d+)%/);
                          if (percentMatch) {
                            scoreText = `${percentMatch[1]}% relevant`;
                          } else {
                            // Nếu có % nhưng không match regex
                            const parts = scoreStr.split('%');
                            if (parts[0]) {
                              scoreText = `${parts[0].trim()}% relevant`;
                            } else {
                              scoreText = "N/A";
                            }
                          }
                        } else {
                          // Nếu là số (0-1), chuyển thành phần trăm
                          const scoreNum = parseFloat(scoreStr);
                          if (!isNaN(scoreNum) && isFinite(scoreNum)) {
                            scoreText = `${Math.round(scoreNum * 100)}% relevant`;
                          } else {
                            scoreText = "N/A";
                          }
                        }
                      } catch (error) {
                        console.error("Error parsing score:", error);
                        scoreText = "N/A";
                      }
                    }
                    
                    return (
                      <li key={i}>
                        {String(source.title || source.file_name || "")} 
                        ({String(source.source || source.source_type || "")})
                        {scoreText && (
                          <span className="confidence">
                            {scoreText}
                          </span>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        </div>
    );
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h3>Welcome to Arbin Chatbot</h3>
            <p>Ask me anything about Arbin Instruments, products, or documentation!</p>
            <div className="example-questions">
              <p>Try asking:</p>
              <ul>
                <li>What battery testing systems does Arbin offer?</li>
                <li>Tell me about Arbin's MITS Pro software</li>
                <li>How do I analyze battery cycle data?</li>
              </ul>
            </div>
          </div>
        ) : (
          messages.map((msg, index) => renderMessage(msg, index))
        )}
        {isLoading && (
          <div className="message bot">
            <div className="message-content">
              <div className="loading">
                <span>●</span>
                <span>●</span>
                <span>●</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}

export default Chat;