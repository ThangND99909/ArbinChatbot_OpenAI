import React, { useState, useEffect } from "react";

const TypingMessage = ({ text, speed = 60 }) => {
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    if (!text) return;

    setDisplayedText("");
    setIsTyping(true);

    let i = 0;
    const interval = setInterval(() => {
      // ✅ Cập nhật an toàn bằng slice (đảm bảo không mất ký tự đầu)
      setDisplayedText(text.slice(0, i + 1));
      i++;

      if (i >= text.length) {
        clearInterval(interval);
        setIsTyping(false);
      }
    }, speed);

    return () => clearInterval(interval);
  }, [text, speed]);

  return (
    <div
      style={{
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        lineHeight: "1.5",
      }}
    >
      {displayedText}
      {isTyping && (
        <span className="chatgpt-typing">
            <span>.</span><span>.</span><span>.</span>
        </span>
        )}
    </div>
  );
};

export default TypingMessage;
