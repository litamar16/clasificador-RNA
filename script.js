function analizarNumero() {
    document.getElementById("resultado").innerHTML =
        "<h2>✅ JAVASCRIPT FUNCIONA</h2><p>El botón Analizar está conectado correctamente.</p>";
}

function limpiar() {
    document.getElementById("numero").value = "";
    document.getElementById("resultado").innerHTML = "";
}
