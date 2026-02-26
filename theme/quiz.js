// Quiz functionality for mdbook
document.addEventListener('DOMContentLoaded', function() {
  const quizzes = document.querySelectorAll('.quiz-container');
  
  quizzes.forEach((quiz, index) => {
    const quizId = `quiz-${index}`;
    quiz.id = quizId;
    
    const buttons = quiz.querySelectorAll('.quiz-option');
    const checkBtn = quiz.querySelector('.quiz-check-btn');
    const feedback = quiz.querySelector('.quiz-feedback');
    
    if (checkBtn) {
      checkBtn.addEventListener('click', function() {
        const selected = quiz.querySelector('input[type="radio"]:checked');
        
        if (!selected) {
          feedback.innerHTML = '<p class="quiz-warning">Please select an answer.</p>';
          return;
        }
        
        const correctAnswer = parseInt(quiz.dataset.correct);
        const selectedAnswer = parseInt(selected.value);
        
        if (selectedAnswer === correctAnswer) {
          feedback.innerHTML = `<div class="quiz-correct"><strong>✓ Correct!</strong></div>`;
          feedback.innerHTML += `<p class="quiz-explanation">${quiz.dataset.explanation}</p>`;
        } else {
          feedback.innerHTML = `<div class="quiz-incorrect"><strong>✗ Incorrect</strong></div>`;
          feedback.innerHTML += `<p class="quiz-explanation">${quiz.dataset.explanation}</p>`;
        }
        
        feedback.style.display = 'block';
      });
    }
  });
});
