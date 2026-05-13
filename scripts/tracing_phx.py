from phoenix.otel import register

# Lanzar Phoenix (se abre en una pestaña nueva)
tracer_provider = register(
    project_name="CRAG-system-tracing",
    auto_instrument=True,
)
tracer = tracer_provider.get_tracer(__name__)
