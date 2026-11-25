const labels = {
  overview:['Research console','Overview'],
  clinical:['Cardiac measurements','Clinical risk'],
  echo:['Imaging workspace','Echo analysis'],
  ecg:['Signal workspace','ECG patterns'],
  mri:['Imaging workspace','Cardiac MRI'],
  valve:['Imaging workspace','Aortic valve']
};

function openView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === id));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === id));
  document.getElementById('sectionLabel').textContent = labels[id][0];
  document.getElementById('pageTitle').textContent = labels[id][1];
  window.scrollTo({top:0,behavior:'smooth'});
}

document.querySelectorAll('.nav-item').forEach(button => button.addEventListener('click', () => openView(button.dataset.view)));
document.querySelectorAll('[data-open]').forEach(element => element.addEventListener('click', () => openView(element.dataset.open)));
document.querySelector('.notice-close').addEventListener('click', () => document.querySelector('.notice').remove());

document.getElementById('estimateRisk').addEventListener('click', () => {
  const age = Number(document.getElementById('age').value || 55);
  const bp = Number(document.getElementById('bp').value || 130);
  const hr = Number(document.getElementById('hr').value || 140);
  const ldl = Number(document.getElementById('ldl').value || 110);
  const hdl = Number(document.getElementById('hdl').value || 50);
  const tg = Number(document.getElementById('tg').value || 150);
  const total = ldl + hdl + tg / 5;
  let score = 0.18 + (age - 20) * 0.006 + (bp - 80) * 0.0012 + (total - 100) * 0.001 - (hr - 80) * 0.0008;
  score = Math.max(.04, Math.min(.96, score));
  const scoreText = document.getElementById('riskScore');
  const ring = document.getElementById('riskRing');
  const target = Math.round(score * 100);
  const start = performance.now();
  const duration = 1100;

  function animateRisk(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(target * eased);
    scoreText.textContent = `${current}%`;
    ring.style.setProperty('--score', `${target * 360 * eased / 100}deg`);

    if (progress < 1) {
      requestAnimationFrame(animateRisk);
    } else {
      scoreText.textContent = `${target}%`;
    }
  }

  requestAnimationFrame(animateRisk);
  document.getElementById('riskCopy').textContent = score >= .5 ? 'Elevated estimate — review alongside clinical context.' : 'Lower estimate — no number replaces clinical assessment.';
});

function previewImage(input, box) {
  input.addEventListener('change', () => {
    if (!input.files[0]) return;
    const old = box.querySelector('img');
    if (old) old.remove();
    const img = document.createElement('img');
    img.src = URL.createObjectURL(input.files[0]);
    box.appendChild(img);
    box.classList.add('has-image');
    updatePair();
  });
}
const edInput = document.getElementById('edInput');
const esInput = document.getElementById('esInput');
previewImage(edInput, document.getElementById('edBox'));
previewImage(esInput, document.getElementById('esBox'));
function updatePair() {
  const count = Number(edInput.files.length > 0) + Number(esInput.files.length > 0);
  document.getElementById('pairStatus').textContent = `${count} of 2 frames`;
  document.getElementById('analyseEcho').disabled = count !== 2;
}
document.getElementById('analyseEcho').addEventListener('click', () => {
  document.getElementById('echoResult').hidden = false;
  document.getElementById('efScore').textContent = '58%';
  document.getElementById('efCopy').textContent = 'Demo estimate — the selected files have not been processed by a model.';
});

document.getElementById('ecgInput').addEventListener('change', event => {
  const file = event.target.files[0];
  if (!file) return;
  document.getElementById('ecgFileName').textContent = file.name;
  const path = [];
  for (let x = 0; x <= 700; x += 5) {
    const y = 110 + Math.sin(x/18)*18 + Math.sin(x/4)*3;
    path.push(`${x === 0?'M':'L'}${x} ${y.toFixed(1)}`);
  }
  document.getElementById('ecgPath').setAttribute('d', path.join(' '));
});
