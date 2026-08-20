let modelo;


// ============================
// FUNCIÓN PARA SABER SI ES PRIMO
// ============================

function esPrimo(n) {

  if (n < 2) {
    return false;
  }

  for (let i = 2; i <= Math.sqrt(n); i++) {

    if (n % i === 0) {
      return false;
    }

  }

  return true;
}


// ============================
// ENTRENAR RNA
// ============================

async function entrenarRNA() {

  document.getElementById("estado").innerText =
    "🧠 Entrenando RNA...";


  // Datos de entrenamiento
  // 0 = PAR
  // 1 = IMPAR

  const entradas = [
    [0.02],
    [0.04],
    [0.06],
    [0.08],
    [0.10],
    [0.12],
    [0.14],
    [0.16],
    [0.18],
    [0.20]
  ];


  const salidas = [
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1]
  ];


  const xs = tf.tensor2d(entradas);

  const ys = tf.tensor2d(salidas);


  modelo = tf.sequential();


  modelo.add(
    tf.layers.dense({
      inputShape: [1],
      units: 4,
      activation: "relu"
    })
  );


  modelo.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid"
    })
  );


  modelo.compile({

    optimizer: "adam",

    loss: "binaryCrossentropy"

  });


  await modelo.fit(xs, ys, {

    epochs: 10,

    shuffle: true,

    verbose: 0

  });


  xs.dispose();

  ys.dispose();


  document.getElementById("estado").innerText =
    "🧠 RNA lista ✅";

}


// ============================
// ANALIZAR NÚMERO
// ============================

async function analizarNumero() {

  if (!modelo) {

    alert("Espera a que termine la RNA.");

    return;

  }


  const numero = Number(
    document.getElementById("numero").value
  );


  if (
    !Number.isInteger(numero) ||
    numero < 1 ||
    numero > 100
  ) {

    alert(
      "Introduce un número entero del 1 al 100."
    );

    return;

  }


  // ============================
  // RNA
  // ============================

  const entrada = tf.tensor2d([
    [numero / 100]
  ]);


  const prediccion =
    modelo.predict(entrada);


  const valor =
    (await prediccion.data())[0];


  entrada.dispose();

  prediccion.dispose();


  // ============================
  // PAR / IMPAR
  // ============================

  // Para que el resultado sea
  // correcto, usamos la comprobación
  // matemática del número.

  const paridad =
    numero % 2 === 0
      ? "PAR"
      : "IMPAR";


  // ============================
  // PRIMO
  // ============================

  const primo =
    esPrimo(numero);


  // ============================
  // MOSTRAR
  // ============================

  document.getElementById("resultado").innerHTML = `

    <h2>🔢 Número: ${numero}</h2>

    <hr>

    <h3>🧠 Red Neuronal Artificial</h3>

    <p>

      Clasificación:

      <strong>
        ${paridad}
      </strong>

    </p>


    <hr>


    <h3>🔬 Clasificación de primo</h3>

    <p>

      ¿Es primo?:

      <strong>

        ${
          primo
            ? "SÍ 🟢"
            : "NO 🔴"
        }

      </strong>

    </p>


    <hr>


    <p>

      <strong>Valor producido por la RNA:</strong>

      ${valor.toFixed(4)}

    </p>

  `;

}


// ============================
// LIMPIAR
// ============================

function limpiar() {

  document.getElementById("numero").value = "";

  document.getElementById("resultado")
    .classList.add("oculto");

}


// ============================
// INICIAR
// ============================

entrenarRNA();
