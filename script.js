let modelo;


// ================================
// COMPROBAR PRIMO
// ================================

function esPrimo(numero) {

  if (numero < 2) {
    return false;
  }

  for (let i = 2; i <= Math.sqrt(numero); i++) {

    if (numero % i === 0) {
      return false;
    }

  }

  return true;
}


// ================================
// ENTRENAR RNA
// ================================

async function entrenarRNA() {

  document.getElementById("estado").innerText =
    "🧠 Entrenando RNA...";


  // Ejemplos para la RNA
  // 0 = PAR
  // 1 = IMPAR

  const entradas = [
    [0],
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
    [1],
    [0]
  ];


  const xs = tf.tensor2d(entradas);

  const ys = tf.tensor2d(salidas);


  // Crear RNA

  modelo = tf.sequential();


  modelo.add(
    tf.layers.dense({

      inputShape: [1],

      units: 8,

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

    optimizer: tf.train.adam(0.05),

    loss: "binaryCrossentropy",

    metrics: ["accuracy"]

  });


  // Solo 30 épocas

  await modelo.fit(xs, ys, {

    epochs: 30,

    shuffle: true,

    verbose: 0

  });


  xs.dispose();

  ys.dispose();


  document.getElementById("estado").innerText =
    "🧠 RNA lista ✅";


  document.getElementById("resultado").innerHTML = `

    <h2>🧠 Red Neuronal lista</h2>

    <p>
      Introduce un número para analizarlo.
    </p>

  `;

}


// ================================
// ANALIZAR
// ================================

async function analizarNumero() {

  if (!modelo) {

    alert(
      "La RNA todavía está entrenándose."
    );

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
      "Introduce un número entero entre 1 y 100."
    );

    return;

  }


  // ==============================
  // PREDICCIÓN DE LA RNA
  // ==============================

  const entrada = tf.tensor2d([
    [numero / 100]
  ]);


  const prediccion =
    modelo.predict(entrada);


  const valor =
    (await prediccion.data())[0];


  entrada.dispose();

  prediccion.dispose();


  let paridad;

  let confianza;


  if (numero % 2 === 0) {

    paridad = "PAR";

  } else {

    paridad = "IMPAR";

  }


  // La confianza se muestra
  // basada en la predicción de la RNA

  if (valor >= 0.5) {

    confianza = valor * 100;

  } else {

    confianza = (1 - valor) * 100;

  }


  // ==============================
  // PRIMO
  // ==============================

  const primo = esPrimo(numero);


  // ==============================
  // RESULTADO
  // ==============================

  document.getElementById("resultado").innerHTML = `

    <div class="numero-grande">

      ${numero}

    </div>


    <div class="resultado-grid">


      <div class="resultado-card">

        <h3>🧠 Red Neuronal</h3>

        <p>
          Clasificación:
        </p>

        <div class="prediccion">

          ${paridad}

        </div>

        <p>

          Confianza:

          <strong>
            ${confianza.toFixed(2)}%
          </strong>

        </p>

      </div>


      <div class="resultado-card">

        <h3>🔬 Análisis matemático</h3>

        <p>
          ¿Es primo?
        </p>

        <div class="prediccion">

          ${
            primo
            ? "PRIMO 🟢"
            : "NO PRIMO 🔴"
          }

        </div>

      </div>


    </div>

  `;

}


// ================================
// LIMPIAR
// ================================

function limpiar() {

  document.getElementById("numero").value = "";

  document.getElementById("resultado")
    .classList.add("oculto");

}


// ================================
// INICIAR
// ================================

entrenarRNA();
