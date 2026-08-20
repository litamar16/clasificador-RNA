let modeloParidad;
let modeloPrimo;


// ==========================================
// COMPROBAR PRIMO
// ==========================================

function esPrimo(numero) {

  if (numero < 2) {
    return false;
  }

  for (
    let i = 2;
    i <= Math.sqrt(numero);
    i++
  ) {

    if (numero % i === 0) {
      return false;
    }

  }

  return true;
}


// ==========================================
// RNA 1
// PAR / IMPAR
// ==========================================

async function entrenarRNAParidad() {

  const entradas = [];
  const salidas = [];


  for (
    let numero = 0;
    numero <= 100;
    numero++
  ) {

    entradas.push([
      numero / 100
    ]);


    salidas.push([
      numero % 2 === 0
        ? 0
        : 1
    ]);

  }


  const xs =
    tf.tensor2d(entradas);

  const ys =
    tf.tensor2d(salidas);


  modeloParidad =
    tf.sequential();


  modeloParidad.add(
    tf.layers.dense({

      inputShape: [1],

      units: 16,

      activation: "relu"

    })
  );


  modeloParidad.add(
    tf.layers.dense({

      units: 8,

      activation: "relu"

    })
  );


  modeloParidad.add(
    tf.layers.dense({

      units: 1,

      activation: "sigmoid"

    })
  );


  modeloParidad.compile({

    optimizer:
      tf.train.adam(0.01),

    loss:
      "binaryCrossentropy",

    metrics:
      ["accuracy"]

  });


  await modeloParidad.fit(
    xs,
    ys,
    {

      epochs: 100,

      shuffle: true,

      verbose: 0

    }
  );


  xs.dispose();

  ys.dispose();

}


// ==========================================
// RNA 2
// PRIMO / NO PRIMO
// ==========================================

async function entrenarRNAPrimos() {

  const entradas = [];

  const salidas = [];


  for (
    let numero = 1;
    numero <= 100;
    numero++
  ) {

    entradas.push([
      numero / 100
    ]);


    salidas.push([
      esPrimo(numero)
        ? 1
        : 0
    ]);

  }


  const xs =
    tf.tensor2d(entradas);

  const ys =
    tf.tensor2d(salidas);


  modeloPrimo =
    tf.sequential();


  modeloPrimo.add(
    tf.layers.dense({

      inputShape: [1],

      units: 32,

      activation: "relu"

    })
  );


  modeloPrimo.add(
    tf.layers.dense({

      units: 16,

      activation: "relu"

    })
  );


  modeloPrimo.add(
    tf.layers.dense({

      units: 8,

      activation: "relu"

    })
  );


  modeloPrimo.add(
    tf.layers.dense({

      units: 1,

      activation: "sigmoid"

    })
  );


  modeloPrimo.compile({

    optimizer:
      tf.train.adam(0.01),

    loss:
      "binaryCrossentropy",

    metrics:
      ["accuracy"]

  });


  await modeloPrimo.fit(
    xs,
    ys,
    {

      epochs: 300,

      shuffle: true,

      verbose: 0

    }
  );


  xs.dispose();

  ys.dispose();

}


// ==========================================
// ENTRENAR TODO
// ==========================================

async function entrenarTodo() {

  try {

    document.getElementById("estado")
      .innerText =
      "Entrenando RNA 1...";

    document.getElementById("progreso")
      .style.width = "20%";

    document.getElementById("textoProgreso")
      .innerText =
      "Aprendiendo a distinguir pares e impares...";


    await entrenarRNAParidad();


    document.getElementById("estado")
      .innerText =
      "Entrenando RNA 2...";

    document.getElementById("progreso")
      .style.width = "55%";

    document.getElementById("textoProgreso")
      .innerText =
      "Aprendiendo a identificar números primos...";


    await entrenarRNAPrimos();


    document.getElementById("progreso")
      .style.width = "100%";


    document.getElementById("estado")
      .innerText =
      "RNA 1 y RNA 2 listas ✅";


    document.getElementById("textoProgreso")
      .innerText =
      "Entrenamiento completado";


    document.getElementById("resultado")
      .classList.remove("oculto");


    document.getElementById("resultado")
      .innerHTML = `

        <div style="text-align:center">

          <div style="font-size:50px">
            🧠
          </div>

          <h2>
            Inteligencia Artificial lista
          </h2>

          <p>
            Las dos Redes Neuronales
            fueron entrenadas correctamente.
          </p>

          <p>
            Puedes introducir un número
            entre <strong>1 y 100</strong>.
          </p>

        </div>

      `;

  }

  catch (error) {

    document.getElementById("estado")
      .innerText =
      "❌ Error al entrenar";

    console.error(error);

    document.getElementById("textoProgreso")
      .innerText =
      "Ocurrió un error. Revisa la consola.";

  }

}


// ==========================================
// ANALIZAR
// ==========================================

async function analizarNumero() {

  if (
    !modeloParidad ||
    !modeloPrimo
  ) {

    alert(
      "Espera a que las RNA terminen de entrenarse."
    );

    return;

  }


  const campo =
    document.getElementById("numero");


  const numero =
    Number(campo.value);


  if (
    campo.value === "" ||
    !Number.isInteger(numero)
  ) {

    alert(
      "Introduce un número entero."
    );

    return;

  }


  if (
    numero < 1 ||
    numero > 100
  ) {

    alert(
      "Introduce un número entre 1 y 100."
    );

    return;

  }


  // ========================================
  // RNA 1
  // ========================================

  const entrada1 =
    tf.tensor2d([
      [numero / 100]
    ]);


  const prediccion1 =
    modeloParidad.predict(
      entrada1
    );


  const valor1 =
    (await prediccion1.data())[0];


  entrada1.dispose();

  prediccion1.dispose();


  let paridad;

  let confianzaParidad;


  if (valor1 >= 0.5) {

    paridad = "IMPAR";

    confianzaParidad =
      valor1 * 100;

  }

  else {

    paridad = "PAR";

    confianzaParidad =
      (1 - valor1) * 100;

  }


  // ========================================
  // RNA 2
  // ========================================

  const entrada2 =
    tf.tensor2d([
      [numero / 100]
    ]);


  const prediccion2 =
    modeloPrimo.predict(
      entrada2
    );


  const valor2 =
    (await prediccion2.data())[0];


  entrada2.dispose();

  prediccion2.dispose();


  let primo;

  let confianzaPrimo;


  if (valor2 >= 0.5) {

    primo = "PRIMO";

    confianzaPrimo =
      valor2 * 100;

  }

  else {

    primo = "NO PRIMO";

    confianzaPrimo =
      (1 - valor2) * 100;

  }


  // ========================================
  // COMPROBACIÓN REAL
  // ========================================

  const comprobacionPar =
    numero % 2 === 0
      ? "PAR"
      : "IMPAR";


  const comprobacionPrimo =
    esPrimo(numero)
      ? "PRIMO"
      : "NO PRIMO";


  // ========================================
  // MOSTRAR
  // ========================================

  const resultado =
    document.getElementById(
      "resultado"
    );


  resultado.classList.remove(
    "oculto"
  );


  resultado.innerHTML = `

    <div class="numero-grande">

      ${numero}

    </div>


    <p style="text-align:center">

      Resultado de las Redes Neuronales

    </p>


    <div class="resultado-grid">


      <!-- RNA 1 -->

      <div class="resultado-card">

        <h3>
          🧠 RNA 1
        </h3>

        <p>
          PAR / IMPAR
        </p>

        <div class="prediccion">

          ${paridad}

        </div>


        <div class="confianza">

          <div class="barra-confianza">

            <div
              style="
                width:${confianzaParidad}%;
              ">
            </div>

          </div>


          <div class="porcentaje">

            ${confianzaParidad.toFixed(2)}%

          </div>

        </div>

      </div>


      <!-- RNA 2 -->

      <div class="resultado-card">

        <h3>
          🧠 RNA 2
        </h3>

        <p>
          PRIMO / NO PRIMO
        </p>

        <div class="prediccion">

          ${primo}

        </div>


        <div class="confianza">

          <div class="barra-confianza">

            <div
              style="
                width:${confianzaPrimo}%;
              ">
            </div>

          </div>


          <div class="porcentaje">

            ${confianzaPrimo.toFixed(2)}%

          </div>

        </div>

      </div>


    </div>


    <!-- COMPROBACIÓN -->

    <div class="comprobacion">

      <h3>
        🔬 Comprobación matemática
      </h3>


      <p>

        Paridad:

        <strong>
          ${comprobacionPar}
        </strong>

      </p>


      <p>

        Número primo:

        <strong>
          ${comprobacionPrimo}
        </strong>

      </p>


    </div>


    <p
      style="
        text-align:center;
        color:#64748b;
        margin-top:20px;
      ">

      Las predicciones anteriores fueron
      realizadas por las Redes Neuronales.

    </p>

  `;

}


// ==========================================
// LIMPIAR
// ==========================================

function limpiar() {

  document.getElementById(
    "numero"
  ).value = "";


  document.getElementById(
    "resultado"
  ).classList.add(
    "oculto"
  );


  document.getElementById(
    "numero"
  ).focus();

}


// ==========================================
// ENTRENAR AL ABRIR LA PÁGINA
// ==========================================

entrenarTodo();
