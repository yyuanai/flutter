const CACHE = 'flutter-v1';
const ASSETS = [
  '/flutter/',
  '/flutter/index.html',
  '/flutter/style.css',
  '/flutter/app.js',
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
});

self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request))
  );
});