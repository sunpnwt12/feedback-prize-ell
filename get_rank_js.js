const scoreRe = new RegExp('^(\\s+)?\\d+\\.\\d+')
const elements = document.querySelectorAll('ul[role="list"] > li[role="listitem"]')
let results = []
let rank = 1;
for (const el of elements) {
    const scoreEls = el.querySelectorAll('p');
    if (scoreEls.length > 0) {
        const scoreEl = scoreEls[0];
        const scoreText = scoreEl.textContent.replaceAll(',', '_');
        if (scoreRe.test(scoreText)) {
            const aEls = el.querySelectorAll('a');
            const spanEls = el.querySelectorAll('span'); // new
            if (aEls.length > 0) {
                const targetA = aEls[aEls.length-1];
                const nbTitle = targetA.textContent.replaceAll(',', '_');
                const targetB = spanEls[spanEls.length-2]; //new
                const nbDesc = targetB.textContent.replaceAll(',', '_'); //new
                results.push([rank.toString(), nbTitle, nbDesc, scoreText].join(',') + ','); // modified
                rank = rank + 1;
            }
        }
    }
}
const csv = results.join('\n');
// console.log(csv);
copy(csv);
console.log('copied to clipboard your scores!');