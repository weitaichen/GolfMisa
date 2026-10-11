// Offline support. Network-first for every app file, so a new deploy is picked up on the
// next visit (HTML, JS and CSS always come from the same version); the cache is only the
// offline fallback. Video byte-range requests are left to the browser.
const VERSION = 'shottracer-v3';
const SHELL = [
  './', 'index.html', 'manifest.webmanifest', 'css/app.css',
  'js/app.js', 'js/analysis.js', 'js/colors.js', 'js/detector.js', 'js/fit.js', 'js/render.js',
  'js/tracker.js', 'js/worker.js',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== location.origin) return;
  if (req.headers.has('range')) return;
  // revalidate with the server every time (a cheap 304 when unchanged): the browser's own
  // HTTP cache must not hand back an old app.js after a deploy
  const fresh = req.mode === 'navigate'
    ? fetch(req.url, { cache: 'no-cache', credentials: 'same-origin' })
    : fetch(new Request(req, { cache: 'no-cache' }));
  e.respondWith(fresh.then((res) => {
    if (res.ok && res.type === 'basic') {
      const copy = res.clone(); // clone before the page consumes the body
      e.waitUntil(caches.open(VERSION).then((c) => c.put(req, copy)));
    }
    return res;
  }).catch(async () => {
    const hit = await caches.match(req, { ignoreSearch: req.mode === 'navigate' });
    if (hit) return hit;
    if (req.mode === 'navigate') return (await caches.match('index.html')) || Response.error();
    return Response.error();
  }));
});
