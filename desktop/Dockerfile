# Builds the browser target only. Electron is never involved.

FROM node:20-alpine AS build

WORKDIR /repo

# Electron would otherwise pull a ~100MB binary this image will never run.
ENV ELECTRON_SKIP_BINARY_DOWNLOAD=1 \
    npm_config_electron_skip_binary_download=1

COPY package.json package-lock.json ./
COPY desktop/package.json ./desktop/
COPY mobile/package.json ./mobile/

# Only the desktop workspace and the root, so Expo's dependency tree
# is never installed.
RUN npm ci --workspace=desktop --include-workspace-root

COPY tsconfig.json ./
COPY shared ./shared
COPY desktop ./desktop

WORKDIR /repo/desktop

# Same-origin API calls through Caddy. Supabase anon key is public by
# design (row-level security is what protects the data), so it is safe
# to bake in. Never pass a service role key here.
ARG VITE_API_BASE_URL=/api
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_SUPABASE_URL=$VITE_SUPABASE_URL \
    VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY

RUN npm run build:web


FROM nginx:1.27-alpine

COPY --from=build /repo/desktop/dist /usr/share/nginx/html
COPY desktop/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
