{{/* The OpenAI-compatible endpoint the gateway sends inference to. */}}
{{- define "llm-routing.engineUrl" -}}
{{- if eq .Values.serving.mode "ray" -}}
http://llm-serve-serve-svc.{{ .Release.Namespace }}.svc.cluster.local:8000
{{- else -}}
http://vllm-serve.{{ .Release.Namespace }}.svc.cluster.local:8000
{{- end -}}
{{- end -}}

{{/* The pods that endpoint resolves to, for the gateway's network policy. */}}
{{- define "llm-routing.engineSelector" -}}
{{- if eq .Values.serving.mode "ray" -}}ray-serve{{- else -}}vllm-serve{{- end -}}
{{- end -}}

{{- define "llm-routing.validate" -}}
{{- if not (has .Values.serving.mode (list "vllm" "ray")) -}}
{{- fail "serving.mode must be vllm or ray" -}}
{{- end -}}
{{- end -}}
