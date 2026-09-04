fetch('api/problems')
  .then((response) => response.json())
  .then((problems) => {
    document.querySelector('#problems').innerHTML = problems
      .map((problem) => `<li><code>${problem.prompt}</code></li>`)
      .join('')
  })
